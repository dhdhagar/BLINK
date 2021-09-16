# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import math
import torch
from torch.utils.data import (DataLoader, SequentialSampler)
import numpy as np
from tqdm import tqdm
import pickle
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from special_partition.special_partition import cluster_linking_partition
from collections import defaultdict
import blink.biencoder.data_process_mult as data_process
import blink.candidate_ranking.utils as utils
from blink.common.params import BlinkParser
from blink.biencoder.biencoder import BiEncoderRanker

from IPython import embed


def get_query_nn(knn,
                 embeds,
                 index,
                 q_embed,
                 searchK=None,
                 gold_idxs=None,
                 type_idx_mapping=None):
    """
    Parameters
    ----------
    knn : int
        the number of nearest-neighbours to return
    embeds : ndarray
        matrix of embeddings
    index : faiss
        faiss index of the embeddings
    q_embed : ndarray
        2-D array containing the query embedding
    searchK: int
        optional parameter, the exact number of nearest-neighbours to retrieve and score
    gold_idxs : array
        optional parameter, list of golden cui indexes
    type_idx_mapping : array
        optional parameter, list mapping type-specific indexes to the indexes of the full dictionary

    Returns
    -------
    nn_idxs : array
        nearest neighbour indices for the query, sorted in descending order of scores
    scores : array
        similarity scores for each nearest neighbour, sorted in descending order
    """
    # To accomodate the approximate-nature of the knn procedure, retrieve more samples and then filter down
    k = searchK if searchK is not None else max(16, 2*knn)

    # Find k nearest neighbours
    _, nn_idxs = index.search(q_embed, k)
    nn_idxs = nn_idxs.astype(np.int64).flatten()
    if type_idx_mapping is not None:
        nn_idxs = type_idx_mapping[nn_idxs]
    nn_embeds = torch.tensor(embeds[nn_idxs]).cuda()

    # Compute query-candidate similarity scores
    scores = torch.flatten(
        torch.mm(torch.tensor(q_embed).cuda(), nn_embeds.T)).cpu()

    # Sort the candidates by descending order of scores
    nn_idxs, scores = zip(
        *sorted(zip(nn_idxs, scores), key=lambda x: -x[1]))

    if gold_idxs is not None:
        # Calculate the knn index at which the gold cui is found (-1 if not found)
        for topk,i in enumerate(nn_idxs):
            if i in gold_idxs:
                break
            topk = -1
        # Return only the top k neighbours, and the recall index
        return np.array(nn_idxs[:knn], dtype=np.int64), np.array(scores[:knn]), topk
    
    # Return only the top k neighbours
    return np.array(nn_idxs[:knn], dtype=np.int64), np.array(scores[:knn])


def partition_graph(graph, n_entities, directed, return_clusters=False):
    """
    Parameters
    ----------
    graph : dict
        object containing rows, cols, data, and shape of the entity-mention joint graph
    n_entities : int
        number of entities in the dictionary
    directed : bool
        whether the graph construction should be directed or undirected
    return_clusters : bool
        flag to indicate if clusters need to be returned from the partition

    Returns
    -------
    partitioned_graph : coo_matrix
        partitioned graph with each mention connected to only one entity
    clusters : dict
        (optional) contains arrays of connected component indices of the graph
    """
    rows, cols, data = cluster_linking_partition(
        graph['rows'],
        graph['cols'],
        graph['data'],
        n_entities,
        directed
    )
    # Construct the partitioned graph
    partitioned_graph = coo_matrix(
        (data, (rows, cols)), shape=graph['shape'])

    if return_clusters:
        # Get an array of the graph with each index marked with the component label that it is connected to
        _, cc_labels = connected_components(
            csgraph=partitioned_graph,
            directed=directed,
            return_labels=True)
        # Store clusters of indices marked with labels with at least 2 connected components
        unique_cc_labels, cc_sizes = np.unique(cc_labels, return_counts=True)
        filtered_labels = unique_cc_labels[cc_sizes > 1]
        clusters = defaultdict(list)
        for i, cc_label in enumerate(cc_labels):
            if cc_label in filtered_labels:
                clusters[cc_label].append(i)
        return partitioned_graph, clusters

    return partitioned_graph


def analyzeClusters(clusters, dictionary, queries, knn):
    """
    Parameters
    ----------
    clusters : dict
        contains arrays of connected component indices of a graph
    dictionary : ndarray
        entity dictionary to evaluate
    queries : ndarray
        mention queries to evaluate
    knn : int
        the number of nearest-neighbour mention candidates considered

    Returns
    -------
    results : dict
        Contains n_entities, n_mentions, knn_mentions, accuracy, failure[], success[]
    """
    n_entities = len(dictionary)
    n_mentions = len(queries)

    results = {
        'n_entities': n_entities,
        'n_mentions': n_mentions,
        'knn_mentions': knn,
        'accuracy': 0,
        'failure': [],
        'success': []
    }
    _debug_n_mens_evaluated, _debug_clusters_wo_entities, _debug_clusters_w_mult_entities = 0, 0, 0

    print("Analyzing clusters...")
    for cluster in clusters.values():
        # The lowest value in the cluster should always be the entity
        pred_entity_idx = cluster[0]
        # Track the graph index of the entity in the cluster
        pred_entity_idxs = [pred_entity_idx]
        if pred_entity_idx >= n_entities:
            # If the first element is a mention, then the cluster does not have an entity
            _debug_clusters_wo_entities += 1
            continue
        pred_entity = dictionary[pred_entity_idx]
        pred_entity_cuis = [str(pred_entity['cui'])]
        _debug_tracked_mult_entities = False
        for i in range(1, len(cluster)):
            men_idx = cluster[i] - n_entities
            if men_idx < 0:
                # If elements after the first are entities, then the cluster has multiple entities
                if not _debug_tracked_mult_entities:
                    _debug_clusters_w_mult_entities += 1
                    _debug_tracked_mult_entities = True
                # Track the graph indices of each entity in the cluster
                pred_entity_idxs.append(cluster[i])
                # Predict based on all entities in the cluster
                pred_entity_cuis += list(set([dictionary[cluster[i]]['cui']]) - set(pred_entity_cuis))
                continue
            _debug_n_mens_evaluated += 1
            men_query = queries[men_idx]
            men_golden_cuis = list(map(str, men_query['label_cuis']))
            report_obj = {
                'mention_id': men_query['mention_id'],
                'mention_name': men_query['mention_name'],
                'mention_gold_cui': '|'.join(men_golden_cuis),
                'mention_gold_cui_name': '|'.join([dictionary[i]['title'] for i in men_query['label_idxs'][:men_query['n_labels']]]),
                'predicted_name': '|'.join([d['title'] for d in [dictionary[i] for i in pred_entity_idxs]]),
                'predicted_cui': '|'.join(pred_entity_cuis),
            }
            # Correct prediction
            if not set(pred_entity_cuis).isdisjoint(men_golden_cuis):
                results['accuracy'] += 1
                results['success'].append(report_obj)
            # Incorrect prediction
            else:
                results['failure'].append(report_obj)
    results['accuracy'] = f"{results['accuracy'] / float(_debug_n_mens_evaluated) * 100} %"
    print(f"Accuracy = {results['accuracy']}")

    # Run sanity checks
    assert n_mentions == _debug_n_mens_evaluated
    assert _debug_clusters_wo_entities == 0
    assert _debug_clusters_w_mult_entities == 0

    return results


def main(params):
    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = utils.get_logger(params["output_path"])

    pickle_src_path = params["pickle_src_path"]
    if pickle_src_path is None or not os.path.exists(pickle_src_path):
        pickle_src_path = output_path

    # Init model
    reranker = BiEncoderRanker(params)
    reranker.model.eval()
    tokenizer = reranker.tokenizer
    model = reranker.model
    device = reranker.device
    n_gpu = reranker.n_gpu

    knn = params["knn"]
    directed_graph = params["directed_graph"]
    use_types = params["use_types"]
    data_split = params["data_split"] # Parameter default is "test"

    # Load test data
    entity_dictionary_loaded = False
    test_dictionary_pkl_path = os.path.join(pickle_src_path, 'test_dictionary.pickle')
    test_tensor_data_pkl_path = os.path.join(pickle_src_path, 'test_tensor_data.pickle')
    test_mention_data_pkl_path = os.path.join(pickle_src_path, 'test_mention_data.pickle')
    if os.path.isfile(test_dictionary_pkl_path):
        print("Loading stored processed entity dictionary...")
        with open(test_dictionary_pkl_path, 'rb') as read_handle:
            test_dictionary = pickle.load(read_handle)
        entity_dictionary_loaded = True
    if os.path.isfile(test_tensor_data_pkl_path) and os.path.isfile(test_mention_data_pkl_path):
        print("Loading stored processed test data...")
        with open(test_tensor_data_pkl_path, 'rb') as read_handle:
            test_tensor_data = pickle.load(read_handle)
        with open(test_mention_data_pkl_path, 'rb') as read_handle:
            mention_data = pickle.load(read_handle)
    else:
        test_samples = utils.read_dataset(data_split, params["data_path"])
        if not entity_dictionary_loaded:
            with open(os.path.join(params["data_path"], 'dictionary.pickle'), 'rb') as read_handle:
                test_dictionary = pickle.load(read_handle)

        # Check if dataset has multiple ground-truth labels
        mult_labels = "labels" in test_samples[0].keys()
        if params["filter_unlabeled"]:
            # Filter samples without gold entities
            test_samples = list(filter(lambda sample: (len(sample["labels"]) > 0) if mult_labels else (sample["label"] is not None), test_samples))
        logger.info("Read %d test samples." % len(test_samples))

        mention_data, test_dictionary, test_tensor_data = data_process.process_mention_data(
            test_samples,
            test_dictionary,
            tokenizer,
            params["max_context_length"],
            params["max_cand_length"],
            multi_label_key="labels" if mult_labels else None,
            context_key=params["context_key"],
            silent=params["silent"],
            logger=logger,
            debug=params["debug"],
            knn=knn,
            dictionary_processed=entity_dictionary_loaded
        )
        print("Saving processed test data...")
        if not entity_dictionary_loaded:
            with open(test_dictionary_pkl_path, 'wb') as write_handle:
                pickle.dump(test_dictionary, write_handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        with open(test_tensor_data_pkl_path, 'wb') as write_handle:
            pickle.dump(test_tensor_data, write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(test_mention_data_pkl_path, 'wb') as write_handle:
            pickle.dump(mention_data, write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    # Store test dictionary token ids
    test_dict_vecs = torch.tensor(
        list(map(lambda x: x['ids'], test_dictionary)), dtype=torch.long)
    # Store test mention token ids
    test_men_vecs = test_tensor_data[:][0]

    n_entities = len(test_dict_vecs)
    n_mentions = len(test_tensor_data)

    # Values of k to run the evaluation against
    knn_vals = [0] + [2**i for i in range(int(math.log(knn, 2)) + 1)]
    # Store the maximum evaluation k
    max_knn = knn_vals[-1]

    # Check if graphs are already built
    graph_path = os.path.join(output_path, 'graphs.pickle')
    if not params['only_recall'] and os.path.isfile(graph_path):
        print("Loading stored joint graphs...")
        with open(graph_path, 'rb') as read_handle:
            joint_graphs = pickle.load(read_handle)
    else:
        # Initialize graphs to store mention-mention and mention-entity similarity score edges;
        # Keyed on k, the number of nearest mentions retrieved
        joint_graphs = {}
        for k in knn_vals:
            joint_graphs[k] = {
                'rows': np.array([]),
                'cols': np.array([]),
                'data': np.array([]),
                'shape': (n_entities+n_mentions, n_entities+n_mentions)
            }

        if use_types:
            print("Dictionary: Embedding and building index")
            dict_embeds, dict_indexes, dict_idxs_by_type = data_process.embed_and_index(reranker, test_dict_vecs, encoder_type="candidate", n_gpu=n_gpu, corpus=test_dictionary, force_exact_search=True)

            # Verifiy embeddings
            og_embeds = torch.load('models/trained/zeshel_og/eval/data_og/cand_encodes.t7')
            world_to_type = {12:'forgotten_realms', 13:'lego', 14:'star_trek', 15:'yugioh'}
            for world in og_embeds:
                for i,oge in enumerate(tqdm(og_embeds[world])):
                    dict_embed_idx = dict_idxs_by_type[world_to_type[world]][i]
                    try:
                        assert torch.eq(oge, torch.tensor(dict_embeds[dict_embed_idx]))
                    except:
                        embed()
                        exit()
            print('PASS')
            exit()
            print("Queries: Embedding and building index")
            men_embeds, men_indexes, men_idxs_by_type = data_process.embed_and_index(reranker, test_men_vecs, encoder_type="context", n_gpu=n_gpu, corpus=mention_data, force_exact_search=True)
        else:
            print("Dictionary: Embedding and building index")
            dict_embeds, dict_index = data_process.embed_and_index(
                reranker, test_dict_vecs, 'candidate', n_gpu=n_gpu)
            print("Queries: Embedding and building index")
            men_embeds, men_index = data_process.embed_and_index(
                reranker, test_men_vecs, 'context', n_gpu=n_gpu)

        recall_accuracy = {2**i: 0 for i in range(int(math.log(params['recall_k'], 2)) + 1)}
        recall_idxs = [0.]*params['recall_k']

        # Find the most similar entity and k-nn mentions for each mention query
        for men_query_idx, men_embed in enumerate(tqdm(men_embeds, total=len(men_embeds), desc="Fetching k-NN")):
            men_embed = np.expand_dims(men_embed, axis=0)
            
            dict_type_idx_mapping, men_type_idx_mapping = None, None
            if use_types:
                entity_type = mention_data[men_query_idx]['type']
                dict_index = dict_indexes[entity_type]
                men_index = men_indexes[entity_type]
                dict_type_idx_mapping = dict_idxs_by_type[entity_type]
                men_type_idx_mapping = men_idxs_by_type[entity_type]
            
            # Fetch nearest entity candidate
            gold_idxs = mention_data[men_query_idx]["label_idxs"][:mention_data[men_query_idx]["n_labels"]]
            dict_cand_idx, dict_cand_score, recall_idx = get_query_nn(
                1, dict_embeds, dict_index, men_embed, searchK=params['recall_k'], gold_idxs=gold_idxs, type_idx_mapping=dict_type_idx_mapping)
            # Compute recall metric
            if recall_idx > -1:
                recall_idxs[recall_idx] += 1.
                for recall_k in recall_accuracy:
                    if recall_idx < recall_k:
                        recall_accuracy[recall_k] += 1.

            if not params['only_recall']:
                # Fetch (k+1) NN mention candidates
                men_cand_idxs, men_cand_scores = get_query_nn(
                    max_knn + 1, men_embeds, men_index, men_embed, type_idx_mapping=men_type_idx_mapping)
                # Filter candidates to remove mention query and keep only the top k candidates
                filter_mask = men_cand_idxs != men_query_idx
                men_cand_idxs, men_cand_scores = men_cand_idxs[filter_mask][:max_knn], men_cand_scores[filter_mask][:max_knn]

                # Add edges to the graphs
                for k in joint_graphs:
                    joint_graph = joint_graphs[k]
                    # Add mention-entity edge
                    joint_graph['rows'] = np.append(
                        joint_graph['rows'], [n_entities+men_query_idx])  # Mentions added at an offset of maximum entities
                    joint_graph['cols'] = np.append(
                        joint_graph['cols'], dict_cand_idx)
                    joint_graph['data'] = np.append(
                        joint_graph['data'], dict_cand_score)
                    if k > 0:
                        # Add mention-mention edges
                        joint_graph['rows'] = np.append(
                            joint_graph['rows'], [n_entities+men_query_idx]*len(men_cand_idxs[:k]))
                        joint_graph['cols'] = np.append(
                            joint_graph['cols'], n_entities+men_cand_idxs[:k])
                        joint_graph['data'] = np.append(
                            joint_graph['data'], men_cand_scores[:k])

        # Compute and print recall metric
        recall_idx_mode = np.argmax(recall_idxs)
        recall_idx_mode_prop = recall_idxs[recall_idx_mode]/np.sum(recall_idxs)
        logger.info(f"""
        Recall metrics (for {len(men_embeds)} queries):
        ---------------""")
        logger.info(f"highest recall idx = {recall_idx_mode} ({recall_idxs[recall_idx_mode]}/{np.sum(recall_idxs)} = {recall_idx_mode_prop})")
        for recall_k in recall_accuracy:
            recall_accuracy[recall_k] /= len(men_embeds)
            logger.info(f"recall@{recall_k} = {recall_accuracy[recall_k]}")

        if params['only_recall']:
            exit()

        # Pickle the graphs
        print("Saving joint graphs...")
        with open(graph_path, 'wb') as write_handle:
            pickle.dump(joint_graphs, write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    results = []
    for k in joint_graphs:
        print(f"\nGraph (k={k}):")
        # Partition graph based on cluster-linking constraints
        partitioned_graph, clusters = partition_graph(
            joint_graphs[k], n_entities, directed_graph, return_clusters=True)
        # Infer predictions from clusters
        result = analyzeClusters(clusters, test_dictionary, mention_data, k)
        # Store result
        results.append(result)

    # Store results
    output_file_name = os.path.join(
        output_path, f"eval_results_{__import__('calendar').timegm(__import__('time').gmtime())}")
    result_overview = {
        'n_entities': results[0]['n_entities'],
        'n_mentions': results[0]['n_mentions'],
        'directed': directed_graph
    }

    try:
        for recall_k in recall_accuracy:
            result_overview[f'recall@{recall_k}'] = recall_accuracy[recall_k]
    except:
        logger.info("Recall data not available since graphs were loaded from disk")
    
    for r in results:
        k = r['knn_mentions']
        result_overview[f'accuracy@knn{k}'] = r['accuracy']
        logger.info(f"accuracy@knn{k} = {r['accuracy']}")
        output_file = f'{output_file_name}-{k}.json'
        with open(output_file, 'w') as f:
            json.dump(r, f, indent=2)
            print(f"\nPredictions @knn{k} saved at: {output_file}")
    with open(f'{output_file_name}.json', 'w') as f:
        json.dump(result_overview, f, indent=2)
        print(f"\nPredictions overview saved at: {output_file_name}.json")


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()
    args = parser.parse_args()
    print(args)
    main(args.__dict__)
