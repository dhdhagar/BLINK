# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import random
import math
import time
import torch
import numpy as np
from tqdm import tqdm
import pickle
import copy

import blink.biencoder.data_process_mult as data_process
import blink.biencoder.eval_cluster_linking as eval_cluster_linking
import blink.candidate_ranking.utils as utils
from blink.common.params import BlinkParser
from blink.biencoder.biencoder import BiEncoderRanker
from blink.crossencoder.crossencoder import CrossEncoderRanker
from blink.crossencoder.train_crossencoder_mst import get_context_doc_ids, get_biencoder_nns, build_cross_concat_input, \
    score_in_batches

from IPython import embed


SCORING_BATCH_SIZE = 64


def load_data(data_split,
              bi_tokenizer,
              max_context_length,
              max_cand_length,
              knn,
              pickle_src_path,
              params,
              logger,
              return_dict_only=False):
    entity_dictionary_loaded = False
    entity_dictionary_pkl_path = os.path.join(pickle_src_path, 'entity_dictionary.pickle')
    if os.path.isfile(entity_dictionary_pkl_path):
        print("Loading stored processed entity dictionary...")
        with open(entity_dictionary_pkl_path, 'rb') as read_handle:
            entity_dictionary = pickle.load(read_handle)
        entity_dictionary_loaded = True

    if return_dict_only and entity_dictionary_loaded:
        return entity_dictionary

    # Load data
    tensor_data_pkl_path = os.path.join(pickle_src_path, f'{data_split}_tensor_data.pickle')
    processed_data_pkl_path = os.path.join(pickle_src_path, f'{data_split}_processed_data.pickle')
    if os.path.isfile(tensor_data_pkl_path) and os.path.isfile(processed_data_pkl_path):
        print("Loading stored processed data...")
        with open(tensor_data_pkl_path, 'rb') as read_handle:
            tensor_data = pickle.load(read_handle)
        with open(processed_data_pkl_path, 'rb') as read_handle:
            processed_data = pickle.load(read_handle)
    else:
        data_samples = utils.read_dataset(data_split, params["data_path"])
        if not entity_dictionary_loaded:
            with open(os.path.join(params["data_path"], 'dictionary.pickle'), 'rb') as read_handle:
                entity_dictionary = pickle.load(read_handle)

        # Check if dataset has multiple ground-truth labels
        mult_labels = "labels" in data_samples[0].keys()
        # Filter samples without gold entities
        data_samples = list(
            filter(lambda sample: (len(sample["labels"]) > 0) if mult_labels else (sample["label"] is not None),
                   data_samples))
        logger.info("Read %d data samples." % len(data_samples))

        processed_data, entity_dictionary, tensor_data = data_process.process_mention_data(
            data_samples,
            entity_dictionary,
            bi_tokenizer,
            max_context_length,
            max_cand_length,
            context_key=params["context_key"],
            multi_label_key="labels" if mult_labels else None,
            silent=params["silent"],
            logger=logger,
            debug=params["debug"],
            knn=knn,
            dictionary_processed=entity_dictionary_loaded
        )
        print("Saving processed data...")
        if not entity_dictionary_loaded:
            with open(entity_dictionary_pkl_path, 'wb') as write_handle:
                pickle.dump(entity_dictionary, write_handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        with open(tensor_data_pkl_path, 'wb') as write_handle:
            pickle.dump(tensor_data, write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(processed_data_pkl_path, 'wb') as write_handle:
            pickle.dump(processed_data, write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    if return_dict_only:
        return entity_dictionary
    return entity_dictionary, tensor_data, processed_data


def filter_by_context_doc_id(mention_idxs, doc_id, doc_id_list, return_numpy=False):
    mask = [doc_id_list[i] == doc_id for i in mention_idxs]
    if isinstance(mention_idxs, list):
        mention_idxs = np.array(mention_idxs)
    mention_idxs = mention_idxs[mask]
    if not return_numpy:
        mention_idxs = list(mention_idxs)
    return mention_idxs, mask


def main(params):
    # Parameter initializations
    logger = utils.get_logger(params["output_path"])
    global SCORING_BATCH_SIZE
    SCORING_BATCH_SIZE = params["scoring_batch_size"]
    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pickle_src_path = params["pickle_src_path"]
    if pickle_src_path is None or not os.path.exists(pickle_src_path):
        pickle_src_path = output_path
    biencoder_indices_path = params["biencoder_indices_path"]
    if biencoder_indices_path is None:
        biencoder_indices_path = output_path
    elif not os.path.exists(biencoder_indices_path):
        os.makedirs(biencoder_indices_path)
    max_k = params["knn"]  # Maximum k-NN graph to build for evaluation
    use_types = params["use_types"]
    within_doc = params["within_doc"]

    # Bi-encoder model
    biencoder_params = copy.deepcopy(params)
    biencoder_params['add_linear'] = False
    bi_reranker = BiEncoderRanker(biencoder_params)
    bi_tokenizer = bi_reranker.tokenizer
    k_biencoder = params["bi_knn"]  # Number of biencoder nearest-neighbors to fetch for cross-encoder scoring (default: 64)

    # Cross-encoder model
    params['add_linear'] = True
    params['add_sigmoid'] = True
    cross_reranker = CrossEncoderRanker(params)
    n_gpu = cross_reranker.n_gpu
    cross_reranker.model.eval()

    # Input lengths
    max_seq_length = params["max_seq_length"]
    max_context_length = params["max_context_length"]
    max_cand_length = params["max_cand_length"]

    # Fix random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cross_reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    data_split = params["data_split"]
    entity_dictionary, tensor_data, processed_data = load_data(data_split,
                                                               bi_tokenizer,
                                                               max_context_length,
                                                               max_cand_length,
                                                               max_k,
                                                               pickle_src_path,
                                                               params,
                                                               logger)
    n_entities = len(entity_dictionary)
    n_mentions = len(processed_data)
    # Store dictionary vectors
    dict_vecs = torch.tensor(list(map(lambda x: x['ids'], entity_dictionary)), dtype=torch.long)
    # Store query vectors
    men_vecs = tensor_data[:][0]

    context_doc_ids = None
    if within_doc:
        # Get context_document_ids for each mention in training and validation
        context_doc_ids = get_context_doc_ids(data_split, params)

    _, biencoder_nns = get_biencoder_nns(bi_reranker=bi_reranker,
                                         biencoder_indices_path=biencoder_indices_path,
                                         entity_dictionary=entity_dictionary,
                                         entity_dict_vecs=dict_vecs,
                                         train_men_vecs=None,
                                         train_processed_data=None,
                                         train_gold_clusters=None,
                                         valid_men_vecs=men_vecs,
                                         valid_processed_data=processed_data,
                                         use_types=use_types,
                                         logger=logger,
                                         n_gpu=n_gpu,
                                         params=params,
                                         train_context_doc_ids=None,
                                         valid_context_doc_ids=context_doc_ids)
    bi_men_idxs = biencoder_nns['men_nns'][:, :k_biencoder]
    bi_ent_idxs = biencoder_nns['dict_nns'][:, :k_biencoder]
    bi_nn_count = np.sum(biencoder_nns['men_nns'] != -1, axis=1)

    # Compute and store the concatenated cross-encoder inputs for validation
    men_concat_inputs, ent_concat_inputs = build_cross_concat_input(biencoder_nns,
                                                                    men_vecs,
                                                                    dict_vecs,
                                                                    max_seq_length,
                                                                    k_biencoder)

    # Values of k to run the evaluation against
    knn_vals = [0] + [2 ** i for i in range(int(math.log(max_k, 2)) + 1)]
    max_k = knn_vals[-1]  # Store the maximum evaluation k
    bi_recall = None

    time_start = time.time()
    # Check if k-NN graphs are already built
    graph_path = os.path.join(output_path, 'graphs.pickle')
    if not params['only_recall'] and os.path.isfile(graph_path):
        print("Loading stored joint graphs...")
        with open(graph_path, 'rb') as read_handle:
            joint_graphs = pickle.load(read_handle)
    else:
        joint_graphs = {}
        for k in knn_vals:
            joint_graphs[k] = {
                'rows': np.array([]),
                'cols': np.array([]),
                'data': np.array([]),
                'shape': (n_entities + n_mentions, n_entities + n_mentions)
            }

        # Score biencoder NNs using cross-encoder
        score_path = os.path.join(output_path, 'cross_scores_indexes.pickle')
        if os.path.isfile(score_path):
            print("Loading stored cross-encoder scores and indexes...")
            with open(score_path, 'rb') as read_handle:
                score_data = pickle.load(read_handle)
            cross_men_topk_idxs = score_data['cross_men_topk_idxs']
            cross_men_topk_scores = score_data['cross_men_topk_scores']
            cross_ent_top1_idx = score_data['cross_ent_top1_idx']
            cross_ent_top1_score = score_data['cross_ent_top1_score']
        else:
            with torch.no_grad():
                logger.info('Eval: Scoring mention-mention edges using cross-encoder...')
                cross_men_scores = score_in_batches(cross_reranker, max_context_length, men_concat_inputs,
                                                    is_context_encoder=True)
                for i in range(len(cross_men_scores)):
                    # Set scores for all invalid nearest neighbours to -infinity (due to variable NN counts of mentions)
                    cross_men_scores[i][bi_nn_count[i]:] = float('-inf')
                cross_men_topk_scores, cross_men_topk_idxs = torch.sort(cross_men_scores, dim=1, descending=True)
                cross_men_topk_idxs = cross_men_topk_idxs.cpu()[:, :max_k]
                cross_men_topk_scores = cross_men_topk_scores.cpu()[:, :max_k]
                logger.info('Eval: Scoring done')

                logger.info('Eval: Scoring mention-entity edges using cross-encoder...')
                cross_ent_scores = score_in_batches(cross_reranker, max_context_length, ent_concat_inputs,
                                                    is_context_encoder=False)
                cross_ent_top1_score, cross_ent_top1_idx = torch.sort(cross_ent_scores, dim=1, descending=True)
                cross_ent_top1_idx = cross_ent_top1_idx.cpu()[:, 0]
                cross_ent_top1_score = cross_ent_top1_score.cpu()[:, 0]
                logger.info('Eval: Scoring done')
            # Pickle the scores and nearest indexes
            print("Saving cross-encoder scores and indexes...")
            with open(score_path, 'wb') as write_handle:
                pickle.dump({
                    'cross_men_topk_idxs': cross_men_topk_idxs,
                    'cross_men_topk_scores': cross_men_topk_scores,
                    'cross_ent_top1_idx': cross_ent_top1_idx,
                    'cross_ent_top1_score': cross_ent_top1_score
                }, write_handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved at: {score_path}")

        # Build k-NN graphs
        bi_recall = 0.
        for men_idx in tqdm(range(len(processed_data)), total=len(processed_data),
                            desc="Eval: Building graphs"):
            # Track biencoder recall@<k_biencoder>
            gold_idx = processed_data[men_idx]["label_idxs"][0]
            if gold_idx in bi_ent_idxs[men_idx]:
                bi_recall += 1.
            # Get nearest entity
            m_e_idx = bi_ent_idxs[men_idx, cross_ent_top1_idx[men_idx]]
            m_e_score = cross_ent_top1_score[men_idx]
            if bi_nn_count[men_idx] > 0:
                # Get nearest mentions
                topk_defined_nn_idxs = cross_men_topk_idxs[men_idx][:bi_nn_count[men_idx]]
                m_m_idxs = bi_men_idxs[
                               men_idx, topk_defined_nn_idxs] + n_entities  # Mentions added at an offset of maximum entities
                m_m_scores = cross_men_topk_scores[men_idx][:bi_nn_count[men_idx]]
            # Add edges to the graphs
            for k in joint_graphs:
                # Add mention-entity edge
                joint_graphs[k]['rows'] = np.append(
                    joint_graphs[k]['rows'], [n_entities + men_idx])  # Mentions added at an offset of maximum entities
                joint_graphs[k]['cols'] = np.append(
                    joint_graphs[k]['cols'], m_e_idx)
                joint_graphs[k]['data'] = np.append(
                    joint_graphs[k]['data'], m_e_score)
                if k > 0 and bi_nn_count[men_idx] > 0:
                    # Add mention-mention edges
                    joint_graphs[k]['rows'] = np.append(
                        joint_graphs[k]['rows'], [n_entities + men_idx] * len(m_m_idxs[:k]))
                    joint_graphs[k]['cols'] = np.append(
                        joint_graphs[k]['cols'], m_m_idxs[:k])
                    joint_graphs[k]['data'] = np.append(
                        joint_graphs[k]['data'], m_m_scores[:k])
        # Pickle the graphs
        print("Saving joint graphs...")
        with open(graph_path, 'wb') as write_handle:
            pickle.dump(joint_graphs, write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved graphs at: {graph_path}")
        # Compute biencoder recall
        bi_recall /= len(processed_data)
        if params['only_recall']:
            logger.info(f"Eval: Biencoder recall@{k_biencoder} = {bi_recall * 100}%")
            exit()

    # Partition graphs and analyze clusters for final predictions
    graph_mode = params.get('graph_mode', None)
    result_overview, results = {
        'n_entities': n_entities,
        'n_mentions': n_mentions
    }, {}
    if graph_mode is None or graph_mode not in ['directed', 'undirected']:
        results['directed'], results['undirected'] = [], []
    else:
        results[graph_mode] = []

    knn_fetch_time = time.time() - time_start
    graph_processing_time = time.time()
    n_graphs_processed = 0.

    for mode in results:
        print(f'\nEvaluation mode: {mode.upper()}')
        for k in joint_graphs:
            if k == 0 and mode == 'undirected' and len(results) > 1:
                continue  # Since @k=0 both modes are equivalent, so skip for one mode
            logger.info(f"\nGraph (k={k}):")
            # Partition graph based on cluster-linking constraints
            partitioned_graph, clusters = eval_cluster_linking.partition_graph(
                joint_graphs[k], n_entities, directed=(mode == 'directed'), return_clusters=True)
            # Infer predictions from clusters
            result = eval_cluster_linking.analyzeClusters(clusters, entity_dictionary, processed_data, k)
            # Store result
            results[mode].append(result)
            n_graphs_processed += 1

    avg_graph_processing_time = (time.time() - graph_processing_time) / n_graphs_processed
    avg_per_graph_time = (knn_fetch_time + avg_graph_processing_time) / 60

    execution_time = (time.time() - time_start) / 60
    # Store results
    output_file_name = os.path.join(
        output_path, f"eval_results_{__import__('calendar').timegm(__import__('time').gmtime())}")

    if bi_recall is not None:
        result_overview[f'biencoder recall@{k_biencoder}'] = f"{bi_recall * 100} %"
    else:
        logger.info("Recall data not available (graphs were loaded from disk)")

    for mode in results:
        mode_results = results[mode]
        result_overview[mode] = {}
        for r in mode_results:
            k = r['knn_mentions']
            result_overview[mode][f'accuracy@knn{k}'] = r['accuracy']
            logger.info(f"{mode} accuracy@knn{k} = {r['accuracy']}")
            output_file = f'{output_file_name}-{mode}-{k}.json'
            with open(output_file, 'w') as f:
                json.dump(r, f, indent=2)
                print(f"\nPredictions ({mode}) @knn{k} saved at: {output_file}")
    with open(f'{output_file_name}.json', 'w') as f:
        json.dump(result_overview, f, indent=2)
        print(f"\nPredictions overview saved at: {output_file_name}.json")

    logger.info("\nThe avg. per graph evaluation time is {} minutes\n".format(avg_per_graph_time))
    logger.info("\nThe total evaluation took {} minutes\n".format(execution_time))


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()
    args = parser.parse_args()
    print(args)
    main(args.__dict__)
