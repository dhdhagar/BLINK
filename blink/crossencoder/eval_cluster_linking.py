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
from sklearn.cluster import KMeans

import blink.biencoder.data_process_mult as data_process
from blink.biencoder.eval_cluster_linking import analyzeClusters, partition_graph
from blink.biencoder.eval_entity_discovery import analyzeClusters as analyze_discovery_clusters
import blink.candidate_ranking.utils as utils
from blink.biencoder.biencoder import BiEncoderRanker
from blink.crossencoder.crossencoder import CrossEncoderRanker
from blink.common.params import BlinkParser
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


def get_entity_idxs_to_drop(processed_data, params, logger):
    print("******** Discovery ********")
    # Proportion of unique entities in the mention set to drop
    ent_drop_prop = params["ent_drop_prop"]
    mention_gold_entities = list(map(lambda x: x['label_idxs'][0], processed_data))
    ents_in_data = np.unique(mention_gold_entities)
    logger.info(f"Dropping {ent_drop_prop * 100}% of {len(ents_in_data)} entities found in mention set")

    # Get entity idxs to drop
    n_ents_dropped = int(ent_drop_prop * len(ents_in_data))
    rng = np.random.default_rng(seed=17)  # Random number generator
    dropped_ent_idxs = rng.choice(ents_in_data, size=n_ents_dropped, replace=False)
    set_dropped_ent_idxs = set(dropped_ent_idxs)

    # Calculate number of mentions without gold entities after dropping
    n_mentions_wo_gold_ents = sum([1 if x in set_dropped_ent_idxs else 0 for x in mention_gold_entities])
    logger.info(f"Dropped {n_ents_dropped} entities")
    logger.info(f"=> Mentions without gold entities = {n_mentions_wo_gold_ents}")
    print("*****************")
    return set_dropped_ent_idxs, n_ents_dropped, n_mentions_wo_gold_ents, mention_gold_entities


def run_inference(entity_dictionary,
                  processed_data,
                  results,
                  result_overview,
                  joint_graphs,
                  n_entities,
                  time_start,
                  output_path,
                  bi_recall,
                  k_biencoder,
                  logger):
    knn_fetch_time = time.time() - time_start
    n_graphs_processed = 0
    graph_processing_time = time.time()
    for mode in results:
        print(f'\nEvaluation mode: {mode.upper()}')
        for k in joint_graphs:
            if k == 0 and mode == 'undirected' and len(results) > 1:
                continue  # Since @k=0 both modes are equivalent, so skip for one mode
            logger.info(f"\nGraph (k={k}):")
            # Partition graph based on cluster-linking constraints
            partitioned_graph, clusters = partition_graph(
                joint_graphs[k], n_entities, directed=(mode == 'directed'), return_clusters=True)
            # Infer predictions from clusters
            result = analyzeClusters(clusters, entity_dictionary, processed_data, k)
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


def run_discovery_experiment(joint_graphs, dropped_ent_idxs, mention_gold_entities, n_entities, n_mentions, data_split,
                             results, output_path, time_start, params, logger):
    def discovery_helper(results, best_result=None, best_config=None, drop_all_entities=False):
        drop_set = dropped_ent_idxs
        if drop_all_entities:
            drop_set = set([i for i in range(n_entities)])
        if exact_threshold is not None:
            thresholds = np.array([0, exact_threshold])
        else:
            thresholds = np.sort(np.concatenate(
                ([0], k_means.fit(joint_graphs[k]['data'].reshape(-1, 1)).cluster_centers_.flatten())))
        for thresh in thresholds:
            print()
            logger.info("Partitioning...")
            logger.info(f"{mode.upper()}, k={k}, threshold={thresh}")
            # Partition graph based on cluster-linking constraints
            partitioned_graph, clusters = partition_graph(
                joint_graphs[k], n_entities, directed=(mode == 'directed'), return_clusters=True,
                exclude=drop_set, threshold=thresh)
            # Analyze cluster against gold clusters
            result = analyze_discovery_clusters(clusters, mention_gold_entities, n_entities, n_mentions, logger)
            results[f'({mode}, {k}, {thresh})'] = result
            if best_result is not None:
                if thresh != 0 and result['average'] > best_result:
                    best_result = result['average']
                    best_config = (mode, k, thresh)
        return best_result, best_config

    graph_mode = ['directed', 'undirected']
    # Number of similarity thresholds to try in order to find best clustering; Default=10
    n_thresholds = params['n_thresholds']
    k_means = KMeans(n_clusters=n_thresholds, random_state=17)
    # Specific threshold to run the experiment with, if the param is passed
    exact_threshold = params.get('exact_threshold', None)
    # Specific value of K defining the number of mention nearest-neighbors used for clustering, if the param is passed
    exact_knn = params.get('exact_knn', None)
    # Store the baseline results
    baselines = {}

    for mode in graph_mode:
        best_result = -1.
        best_config = None
        for k in joint_graphs:
            if k == 0:
                continue
            # First run the baseline (i.e. with no entity edges), which stores results in the passed arg
            logger.info("Baseline:")
            discovery_helper(baselines, drop_all_entities=True)
            if (exact_knn is None and 0 < k <= params["knn"]) or (exact_knn is not None and k == exact_knn):
                best_result, best_config = discovery_helper(results, best_result, best_config)
        results[f'best_{mode}_config'] = best_config
        results[f'best_{mode}_result'] = best_result
        results['baselines'] = baselines

    # Store results
    output_file_name = os.path.join(
        output_path, f"{data_split}_eval_discovery_{__import__('calendar').timegm(__import__('time').gmtime())}.json")

    with open(output_file_name, 'w') as f:
        json.dump(results, f, indent=2)
        print(f"\nAnalysis saved at: {output_file_name}")
    execution_time = (time.time() - time_start) / 60
    logger.info(f"\nTotal time taken: {execution_time} minutes\n")


def save_topk_biencoder_cands(bi_reranker,
                             use_types, logger, n_gpu, params,
                             bi_tokenizer, max_context_length,
                             max_cand_length, pickle_src_path, topk=64):
    entity_dictionary = load_data('train',
                                  bi_tokenizer,
                                  max_context_length,
                                  max_cand_length,
                                  1,
                                  pickle_src_path,
                                  params,
                                  logger,
                                  return_dict_only=True)
    entity_dict_vecs = torch.tensor(list(map(lambda x: x['ids'], entity_dictionary)), dtype=torch.long)

    logger.info('Biencoder: Embedding and indexing entity dictionary')
    if use_types:
        _, dict_indexes, dict_idxs_by_type = data_process.embed_and_index(
            bi_reranker, entity_dict_vecs, encoder_type="candidate", n_gpu=n_gpu, corpus=entity_dictionary,
            force_exact_search=True, batch_size=params['embed_batch_size'])
    else:
        _, dict_index = data_process.embed_and_index(bi_reranker, entity_dict_vecs,
                                                     encoder_type="candidate", n_gpu=n_gpu,
                                                     force_exact_search=True,
                                                     batch_size=params['embed_batch_size'])
    logger.info('Biencoder: Embedding and indexing finished')

    for mode in ["train", "valid", "test"]:
        logger.info(f"Biencoder: Fetching top-{topk} biencoder candidates for {mode} set")
        _, tensor_data, processed_data = load_data(mode,
                                                   bi_tokenizer,
                                                   max_context_length,
                                                   max_cand_length,
                                                   1,
                                                   pickle_src_path,
                                                   params,
                                                   logger)
        men_vecs = tensor_data[:][0]

        logger.info('Biencoder: Embedding mention data')
        if use_types:
            men_embeddings, _, men_idxs_by_type = data_process.embed_and_index(
                bi_reranker, men_vecs, encoder_type="context", n_gpu=n_gpu, corpus=processed_data,
                force_exact_search=True, batch_size=params['embed_batch_size'])
        else:
            men_embeddings = data_process.embed_and_index(bi_reranker, men_vecs,
                                                                encoder_type="context", n_gpu=n_gpu,
                                                                force_exact_search=True,
                                                                batch_size=params['embed_batch_size'],
                                                                only_embed=True)
        logger.info('Biencoder: Embedding finished')

        logger.info("Biencoder: Finding nearest entities for each mention...")
        if not use_types:
            _, bi_dict_nns = dict_index.search(men_embeddings, topk)
        else:
            bi_dict_nns = np.zeros((len(men_embeddings), topk), dtype=int)
            for entity_type in dict_indexes:
                men_embeds_by_type = men_embeddings[men_idxs_by_type[entity_type]]
                _, dict_nns_by_type = dict_indexes[entity_type].search(men_embeds_by_type, topk)
                dict_nns_idxs = np.array(list(map(lambda x: dict_idxs_by_type[entity_type][x], dict_nns_by_type)))
                for i, idx in enumerate(men_idxs_by_type[entity_type]):
                    bi_dict_nns[idx] = dict_nns_idxs[i]
        logger.info("Biencoder: Search finished")

        labels = [-1]*len(bi_dict_nns)
        for men_idx in range(len(bi_dict_nns)):
            gold_idx = processed_data[men_idx]["label_idxs"][0]
            for i in range(len(bi_dict_nns[men_idx])):
                if bi_dict_nns[men_idx][i] == gold_idx:
                    labels[men_idx] = i
                    break

        logger.info(f"Biencoder: Saving top-{topk} biencoder candidates for {mode} set")
        save_data_path = os.path.join(params['output_path'], f'candidates_{mode}_top{topk}.t7')
        torch.save({
            "mode": mode,
            "candidates": bi_dict_nns,
            "labels": labels
        }, save_data_path)
        logger.info("Biencoder: Saved")


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
    discovery_mode = params["discovery"]

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

    # The below code is to generate the candidates for cross-encoder training and inference
    if params["save_topk_result"]:
        save_topk_biencoder_cands(bi_reranker,
                                  use_types, logger, n_gpu, params,
                                  bi_tokenizer, max_context_length,
                                  max_cand_length, pickle_src_path, topk=64)
        exit()

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

    discovery_entities = []
    if discovery_mode:
        discovery_entities, n_ents_dropped, n_mentions_wo_gold_ents, mention_gold_entities = get_entity_idxs_to_drop(
            processed_data, params,
            logger)

    context_doc_ids = None
    if within_doc:
        # Get context_document_ids for each mention in training and validation
        context_doc_ids = get_context_doc_ids(data_split, params)
    params["only_evaluate"] = True  # Needed to call get_biencoder_nns() correctly
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
                                                    is_context_encoder=True, scoring_batch_size=SCORING_BATCH_SIZE)
                for i in range(len(cross_men_scores)):
                    # Set scores for all invalid nearest neighbours to -infinity (due to variable NN counts of mentions)
                    cross_men_scores[i][bi_nn_count[i]:] = float('-inf')
                cross_men_topk_scores, cross_men_topk_idxs = torch.sort(cross_men_scores, dim=1, descending=True)
                cross_men_topk_idxs = cross_men_topk_idxs.cpu()[:, :max_k]
                cross_men_topk_scores = cross_men_topk_scores.cpu()[:, :max_k]
                logger.info('Eval: Scoring done')

                logger.info('Eval: Scoring mention-entity edges using cross-encoder...')
                cross_ent_scores = score_in_batches(cross_reranker, max_context_length, ent_concat_inputs,
                                                    is_context_encoder=False, scoring_batch_size=SCORING_BATCH_SIZE)
                cross_ent_top1_score, cross_ent_top1_idx = torch.sort(cross_ent_scores, dim=1, descending=True)
                cross_ent_top1_idx = cross_ent_top1_idx.cpu()
                cross_ent_top1_score = cross_ent_top1_score.cpu()
                if discovery_mode:
                    # Replace the first value in each row with an entity not in the drop set
                    for i in range(cross_ent_top1_idx.shape[0]):
                        for j in range(cross_ent_top1_idx.shape[1]):
                            if cross_ent_top1_idx[i, j] not in discovery_entities:
                                cross_ent_top1_idx[i, 0] = cross_ent_top1_idx[i, j]
                                cross_ent_top1_score[i, 0] = cross_ent_top1_score[i, j]
                                break
                cross_ent_top1_idx = cross_ent_top1_idx[:, 0]
                cross_ent_top1_score = cross_ent_top1_score[:, 0]
                logger.info('Eval: Scoring done')
            # Pickle the scores and nearest indexes
            logger.info("Saving cross-encoder scores and indexes...")
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

    graph_mode = params.get('graph_mode', None)

    if discovery_mode:
        # Run the entity discovery experiment
        results = {
            'data_split': data_split.upper(),
            'n_entities': n_entities,
            'n_mentions': n_mentions,
            'n_entities_dropped': f"{n_ents_dropped} ({params['ent_drop_prop'] * 100}%)",
            'n_mentions_wo_gold_entities': n_mentions_wo_gold_ents
        }
        run_discovery_experiment(joint_graphs,
                                 discovery_entities,
                                 mention_gold_entities,
                                 n_entities,
                                 n_mentions,
                                 data_split,
                                 results,
                                 output_path,
                                 time_start,
                                 params,
                                 logger)
    else:
        # Run entity linking inference
        result_overview, results = {
                                       'n_entities': n_entities,
                                       'n_mentions': n_mentions
                                   }, {}
        if graph_mode is None or graph_mode not in ['directed', 'undirected']:
            results['directed'], results['undirected'] = [], []
        else:
            results[graph_mode] = []
        run_inference(entity_dictionary,
                      processed_data,
                      results,
                      result_overview,
                      joint_graphs,
                      n_entities,
                      time_start,
                      output_path,
                      bi_recall,
                      k_biencoder,
                      logger)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()
    parser.add_joint_train_args()
    args = parser.parse_args()
    print(args)
    main(args.__dict__)
