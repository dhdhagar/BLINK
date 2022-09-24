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
import time
import torch
import csv

from torch.utils.data import (DataLoader, SequentialSampler)
import numpy as np
from tqdm import tqdm
import pickle
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans

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


def partition_graph(graph, n_entities, directed, return_clusters=False, exclude=set(), threshold=None):
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
    exclude : set
        vertices (both rows and columns) whose edges should be dropped in the graph
    threshold : float
        similarity value below which edges should be dropped


    Returns
    -------
    partitioned_graph : coo_matrix
        partitioned graph with each mention connected to only one entity
    clusters : dict
        (optional) contains arrays of connected component indices of the graph
    """
    rows, cols, data, shape = graph['rows'], graph['cols'], graph['data'], graph['shape']

    rows, cols, data = cluster_linking_partition(
        rows,
        cols,
        data,
        n_entities,
        directed,
        exclude=exclude,
        threshold=threshold
    )
    # Construct the partitioned graph
    partitioned_graph = coo_matrix(
        (data, (rows, cols)), shape=shape)

    if return_clusters:
        # Get an array of the graph with each index marked with the component label that it is connected to
        _, cc_labels = connected_components(
            csgraph=partitioned_graph,
            directed=directed,
            return_labels=True)
        # Store clusters of indices marked with labels with at least 2 connected components
        unique_cc_labels, cc_sizes = np.unique(cc_labels, return_counts=True)
        filtered_labels = unique_cc_labels[cc_sizes >= 2]
        clusters = defaultdict(list)
        for i, cc_label in enumerate(cc_labels):
            if cc_label in filtered_labels:
                clusters[cc_label].append(i)
        return partitioned_graph, clusters

    return partitioned_graph


def analyzeClusters(clusters, dictionary, queries, knn, n_train_mentions=0):
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
            men_idx -= n_train_mentions
            if men_idx < 0:
                # Query is from train set
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


def filter_by_context_doc_id(mention_idxs, doc_id, doc_id_list, return_numpy=False):
    mask = [doc_id_list[i] == doc_id for i in mention_idxs]
    if isinstance(mention_idxs, list):
        mention_idxs = np.array(mention_idxs)
    mention_idxs = mention_idxs[mask]
    if not return_numpy:
        mention_idxs = list(mention_idxs)
    return mention_idxs, mask


def read_data(split, params, logger):
    samples = utils.read_dataset(split, params["data_path"])
    # Check if dataset has multiple ground-truth labels
    has_mult_labels = "labels" in samples[0].keys()
    if params["filter_unlabeled"]:
        # Filter samples without gold entities
        samples = list(
            filter(lambda sample: (len(sample["labels"]) > 0) if has_mult_labels else (sample["label"] is not None),
                   samples))
    logger.info(f"Read {len(samples)} {split} samples.")
    return samples, has_mult_labels


def solve_gaussian(m1, m2, std1, std2):
    # Below based on https://stackoverflow.com/a/22579904/2923106
    # a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    # b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    # c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)
    # return np.roots([a, b, c])

    # Below based on https://stats.stackexchange.com/a/571886
    minus_b = (m1 * std2 ** 2 - m2 * std1 ** 2)
    root_b2_minus_4ac = std1 * std2 * np.sqrt((m1 - m2) ** 2 + (std2 ** 2 - std1 ** 2) * np.log(std2 ** 2 / std1 ** 2))
    two_a = std2 ** 2 - std1 ** 2
    roots = [(minus_b - root_b2_minus_4ac) / two_a, (minus_b + root_b2_minus_4ac) / two_a]
    return roots


def main(params):
    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = utils.get_logger(params["output_path"], 'log-eval')

    pickle_src_path = params["pickle_src_path"]
    if pickle_src_path is None or not os.path.exists(pickle_src_path):
        pickle_src_path = output_path

    embed_data_path = params["embed_data_path"]
    if embed_data_path is None or not os.path.exists(embed_data_path):
        embed_data_path = output_path

    cands_path = os.path.join(output_path, 'cands.pickle')

    # Init model
    reranker = BiEncoderRanker(params)
    reranker.model.eval()
    tokenizer = reranker.tokenizer
    n_gpu = reranker.n_gpu

    knn = params["knn"]
    use_types = params["use_types"]
    within_doc = params["within_doc"]
    data_split = params["data_split"] # Default = "test"

    # Load test data
    test_samples = None
    entity_dictionary_loaded = False
    test_dictionary_pkl_path = os.path.join(pickle_src_path, 'test_dictionary.pickle')
    test_tensor_data_pkl_path = os.path.join(pickle_src_path, 'test_tensor_data.pickle')
    test_mention_data_pkl_path = os.path.join(pickle_src_path, 'test_mention_data.pickle')
    if params['transductive']:
        train_tensor_data_pkl_path = os.path.join(pickle_src_path, 'train_tensor_data.pickle')
        train_mention_data_pkl_path = os.path.join(pickle_src_path, 'train_mention_data.pickle')
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
            entity_dictionary_loaded = True
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

    if within_doc:
        if test_samples is None:
            test_samples, _ = read_data(data_split, params, logger)
        test_context_doc_ids = [s['context_doc_id'] for s in test_samples]

    if params["transductive"]:
        if os.path.isfile(train_tensor_data_pkl_path) and os.path.isfile(train_mention_data_pkl_path):
            print("Loading stored processed train data...")
            with open(train_tensor_data_pkl_path, 'rb') as read_handle:
                train_tensor_data = pickle.load(read_handle)
            with open(train_mention_data_pkl_path, 'rb') as read_handle:
                train_mention_data = pickle.load(read_handle)
        else:
            train_samples = utils.read_dataset('train', params["data_path"])

            # Check if dataset has multiple ground-truth labels
            mult_labels = "labels" in train_samples[0].keys()
            logger.info("Read %d test samples." % len(test_samples))

            train_mention_data, _, train_tensor_data = data_process.process_mention_data(
                train_samples,
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
            print("Saving processed train data...")
            with open(train_tensor_data_pkl_path, 'wb') as write_handle:
                pickle.dump(train_tensor_data, write_handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            with open(train_mention_data_pkl_path, 'wb') as write_handle:
                pickle.dump(train_mention_data, write_handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        # Store train mention token ids
        train_men_vecs = train_tensor_data[:][0]
        n_mentions += len(train_tensor_data)
        n_train_mentions = len(train_tensor_data)

    # if os.path.isfile(cands_path):
    #     print("Loading stored candidates...")
    #     with open(cands_path, 'rb') as read_handle:
    #         men_cands = pickle.load(read_handle)
    # else:
    # Check and load stored embedding data
    embed_data_path = os.path.join(embed_data_path, 'embed_data.t7')
    embed_data = None
    if os.path.isfile(embed_data_path):
        embed_data = torch.load(embed_data_path)

    if use_types:
        if embed_data is not None:
            logger.info('Loading stored embeddings and computing indexes')
            dict_embeds = embed_data['dict_embeds']
            if 'dict_idxs_by_type' in embed_data:
                dict_idxs_by_type = embed_data['dict_idxs_by_type']
            else:
                dict_idxs_by_type = data_process.get_idxs_by_type(test_dictionary)
            dict_indexes = data_process.get_index_from_embeds(dict_embeds, dict_idxs_by_type,
                                                              force_exact_search=params['force_exact_search'],
                                                              probe_mult_factor=params['probe_mult_factor'])
            men_embeds = embed_data['men_embeds']
            if 'men_idxs_by_type' in embed_data:
                men_idxs_by_type = embed_data['men_idxs_by_type']
            else:
                men_idxs_by_type = data_process.get_idxs_by_type(mention_data)
            men_indexes = data_process.get_index_from_embeds(men_embeds, men_idxs_by_type,
                                                             force_exact_search=params['force_exact_search'],
                                                             probe_mult_factor=params['probe_mult_factor'])
        else:
            logger.info("Dictionary: Embedding and building index")
            dict_embeds, dict_indexes, dict_idxs_by_type = data_process.embed_and_index(reranker, test_dict_vecs,
                                                                                        encoder_type="candidate",
                                                                                        n_gpu=n_gpu,
                                                                                        corpus=test_dictionary,
                                                                                        force_exact_search=params[
                                                                                            'force_exact_search'],
                                                                                        batch_size=params[
                                                                                            'embed_batch_size'],
                                                                                        probe_mult_factor=params[
                                                                                            'probe_mult_factor'])
            logger.info("Queries: Embedding and building index")
            vecs = test_men_vecs
            men_data = mention_data
            if params['transductive']:
                vecs = torch.cat((train_men_vecs, vecs), dim=0)
                men_data = train_mention_data + mention_data
            men_embeds, men_indexes, men_idxs_by_type = data_process.embed_and_index(reranker, vecs,
                                                                                     encoder_type="context",
                                                                                     n_gpu=n_gpu, corpus=men_data,
                                                                                     force_exact_search=params[
                                                                                         'force_exact_search'],
                                                                                     batch_size=params[
                                                                                         'embed_batch_size'],
                                                                                     probe_mult_factor=params[
                                                                                         'probe_mult_factor'])
    else:
        if embed_data is not None:
            logger.info('Loading stored embeddings and computing indexes')
            dict_embeds = embed_data['dict_embeds']
            dict_index = data_process.get_index_from_embeds(dict_embeds,
                                                            force_exact_search=params['force_exact_search'],
                                                            probe_mult_factor=params['probe_mult_factor'])
            men_embeds = embed_data['men_embeds']
            men_index = data_process.get_index_from_embeds(men_embeds,
                                                           force_exact_search=params['force_exact_search'],
                                                           probe_mult_factor=params['probe_mult_factor'])
        else:
            logger.info("Dictionary: Embedding and building index")
            dict_embeds, dict_index = data_process.embed_and_index(
                reranker, test_dict_vecs, 'candidate', n_gpu=n_gpu, force_exact_search=params['force_exact_search'],
                batch_size=params['embed_batch_size'], probe_mult_factor=params['probe_mult_factor'])
            logger.info("Queries: Embedding and building index")
            vecs = test_men_vecs
            if params['transductive']:
                vecs = torch.cat((train_men_vecs, vecs), dim=0)
            men_embeds, men_index = data_process.embed_and_index(
                reranker, vecs, 'context', n_gpu=n_gpu, force_exact_search=params['force_exact_search'],
                batch_size=params['embed_batch_size'], probe_mult_factor=params['probe_mult_factor'])

    # Save computed embedding data if not loaded from disk
    if embed_data is None:
        embed_data = {}
        embed_data['dict_embeds'] = dict_embeds
        embed_data['men_embeds'] = men_embeds
        if use_types:
            embed_data['dict_idxs_by_type'] = dict_idxs_by_type
            embed_data['men_idxs_by_type'] = men_idxs_by_type
        # NOTE: Cannot pickle faiss index because it is a SwigPyObject
        torch.save(embed_data, embed_data_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Starting KNN search...")
    # Fetch (k+1) NN mention candidates
    if not use_types:
        _men_embeds = men_embeds
        if params['transductive']:
            _men_embeds = _men_embeds[n_train_mentions:]
        n_mens_to_fetch = len(_men_embeds) if within_doc else knn+1
        nn_men_dists, nn_men_idxs = men_index.search(_men_embeds, n_mens_to_fetch)
    else:
        query_len = len(men_embeds) - (n_train_mentions if params['transductive'] else 0)
        nn_men_idxs = -1 * np.ones((query_len, query_len), dtype=int)
        nn_men_dists = -1 * np.ones((query_len, query_len), dtype='float64')
        for entity_type in men_indexes:
            men_embeds_by_type = men_embeds[
                men_idxs_by_type[entity_type][men_idxs_by_type[entity_type] >= n_train_mentions]] if params[
                'transductive'] else men_embeds[men_idxs_by_type[entity_type]]
            n_mens_to_fetch = len(men_embeds_by_type) if within_doc else knn + 1
            nn_men_dists_by_type, nn_men_idxs_by_type = men_indexes[entity_type].search(men_embeds_by_type,
                                                                                        min(n_mens_to_fetch,
                                                                                        len(men_embeds_by_type)))
            nn_men_idxs_by_type = np.array(
                list(map(lambda x: men_idxs_by_type[entity_type][x], nn_men_idxs_by_type)))
            i = -1
            for idx in men_idxs_by_type[entity_type]:
                if params['transductive']:
                    idx -= n_train_mentions
                if idx < 0:
                    continue
                i += 1
                nn_men_idxs[idx][:len(nn_men_idxs_by_type[i])] = nn_men_idxs_by_type[i]
                nn_men_dists[idx][:len(nn_men_dists_by_type[i])] = nn_men_dists_by_type[i]
    logger.info("Search finished")

    logger.info(f'Calculating mention recall@{knn}')
    # Find the most similar entity and k-nn mentions for each mention query
    men_recall_knn = []
    men_cands = []
    men_scores = []
    cui_sums = defaultdict(int)
    for m in mention_data:
        cui_sums[m['label_cuis'][0]] += 1
    for idx in range(len(nn_men_idxs)):
        filter_mask_neg1 = nn_men_idxs[idx] != -1
        men_cand_idxs = nn_men_idxs[idx][filter_mask_neg1]
        men_cand_scores = nn_men_dists[idx][filter_mask_neg1]

        if within_doc:
            men_cand_idxs, wd_mask = filter_by_context_doc_id(men_cand_idxs,
                                                              test_context_doc_ids[idx],
                                                              test_context_doc_ids, return_numpy=True)
            men_cand_scores = men_cand_scores[wd_mask]

        # Filter candidates to remove mention query and keep only the top k candidates
        filter_mask = men_cand_idxs != idx
        men_cand_idxs, men_cand_scores = men_cand_idxs[filter_mask][:knn], men_cand_scores[filter_mask][:knn]
        assert len(men_cand_idxs) == knn
        men_cands.append(men_cand_idxs)
        men_scores.append(men_cand_scores)
        # Calculate mention recall@k
        gold_cui = mention_data[idx]['label_cuis'][0]
        if cui_sums[gold_cui] > 1:
            recall_hit = [1. for midx in men_cand_idxs if mention_data[midx]['label_cuis'][0] == gold_cui]
            recall_knn = sum(recall_hit) / min(cui_sums[gold_cui] - 1, knn)
            men_recall_knn.append(recall_knn)
    logger.info('Done')



    # Compute optimal score threshold to use for pairwise similarity score metric
    if params["compute_pairwise_acc_threshold"]:
        logger.info("Computing optimal threshold for pairwise accuracy (F1)")
        threshold_range = [5, 10, 15, 20, 25, 30]  # [2**i for i in range(int(math.log(params['n_thresholds'], 2)) + 1)]
        knn_range = [2**i for i in range(int(math.log(knn, 2)) + 1)]
        for n_thresholds in threshold_range:
            # n_thresholds = params['n_thresholds']  # Default is 10
            k_means = KMeans(n_clusters=n_thresholds, random_state=17)
            for _knn in knn_range:
                men_scores_knn = [m[:_knn] for m in men_scores]
                men_cands_knn = [m[:_knn] for m in men_cands]
                thresholds = np.sort(
                    np.concatenate(([0], k_means.fit(np.array(men_scores_knn).flatten().reshape(-1, 1)).cluster_centers_.flatten())))
                n_gold_edges = sum([cui_sums[cui] * (cui_sums[cui] - 1) / 2 for cui in cui_sums])
                best_f1 = -float('inf')
                best_thresh, best_knn, best_n_thresh = None, None, None
                for thresh in thresholds:
                    print()
                    logger.info(f"Running with n_thresh={n_thresholds}, knn={_knn}, thresh={thresh}")
                    seen = set()
                    n_correct_edges = 0.
                    n_retained_edges = 0.
                    for idx in range(len(men_cands_knn)):
                        gold_cui = mention_data[idx]['label_cuis'][0]
                        for nn_i, nn_score in enumerate(men_scores_knn[idx]):
                            if nn_score > thresh:
                                # Valid edge
                                nn_idx = men_cands_knn[idx][nn_i]
                                if (idx, nn_idx) in seen or (nn_idx, idx) in seen:
                                    continue
                                seen.add((idx, nn_idx))
                                n_retained_edges += 1
                                nn_gold_cui = mention_data[nn_idx]['label_cuis'][0]
                                if nn_gold_cui == gold_cui:
                                    n_correct_edges += 1
                    if n_retained_edges == 0:
                        continue
                    precision = n_correct_edges / n_retained_edges
                    recall = n_correct_edges / n_gold_edges
                    if precision == 0 and recall == 0:
                        continue
                    f1 = (2 * precision * recall) / (precision + recall)
                    logger.info(f"n_correct={n_correct_edges}, n_retained={n_retained_edges}, n_gold={n_gold_edges}")
                    logger.info(f"f1={f1}, recall={recall}")
                    if f1 > best_f1:
                        logger.info(f"New best F1={f1} @ thresh={thresh}")
                        best_f1 = f1
                        best_thresh = thresh
                        best_knn = _knn
                        best_n_thresh = n_thresholds
        print("\n")
        logger.info(f"Best F1 = {best_f1} @ threshold = {best_thresh}")
        logger.info(f"Best knn = {best_knn}, best n_threshold = {best_n_thresh}")
        exit(0)

    if params["gen_correlation_clustering_data"]:
        # Take exact threshold
        threshold = params["exact_threshold"]
        logger.info(f"Dropping edges using threshold={threshold}")
        # Drop edges below that threshold
        seen = set()
        retained_edges = []
        n_dropped = 0
        n_correct_edges = 0.
        n_gold_edges = sum([cui_sums[cui] * (cui_sums[cui] - 1) / 2 for cui in cui_sums])
        for idx in range(len(men_cands)):
            gold_cui = mention_data[idx]['label_cuis'][0]
            for nn_i, nn_score in enumerate(men_scores[idx]):
                if not params["normalize_correlation_clustering_data"]:
                    if nn_score > threshold:
                        # Valid edge
                        nn_idx = men_cands[idx][nn_i]
                        if (idx, nn_idx) in seen or (nn_idx, idx) in seen:
                            continue
                        seen.add((idx, nn_idx))
                        retained_edges.append({
                            "from": idx,
                            "to": nn_idx,
                            "weight": nn_score
                        })
                        nn_gold_cui = mention_data[nn_idx]['label_cuis'][0]
                        if nn_gold_cui == gold_cui:
                            n_correct_edges += 1
                    else:
                        n_dropped += 1
                else:
                    nn_idx = men_cands[idx][nn_i]
                    if (idx, nn_idx) in seen or (nn_idx, idx) in seen:
                        continue
                    seen.add((idx, nn_idx))
                    norm_score = nn_score - threshold
                    if norm_score == 0:
                        n_dropped += 1
                        continue
                    retained_edges.append({
                        "from": idx,
                        "to": nn_idx,
                        "weight": norm_score
                    })
                    nn_gold_cui = mention_data[nn_idx]['label_cuis'][0]
                    if norm_score > 0 and nn_gold_cui == gold_cui:
                        n_correct_edges += 1
        logger.info(f"Dropped {n_dropped} edges, retained {len(retained_edges)} edges")
        # Calculate F1 of positive edges
        precision = n_correct_edges / len(retained_edges)
        recall = n_correct_edges / n_gold_edges
        if precision == 0 and recall == 0:
            logger.info("Error: Precision and recall both zero! Exiting...")
            exit(0)
        f1 = (2 * precision * recall) / (precision + recall)
        logger.info(f"n_correct={n_correct_edges}, n_retained={len(retained_edges)}, n_gold={n_gold_edges}")
        logger.info(f"f1={f1}, recall={recall}")
        # Generate tsv in the expected format and dump
        with open(f'{output_path}/correlation_data.tsv', mode='w') as fh:
            csv_writer = csv.writer(fh, delimiter='\t')
            csv_writer.writerow([n_mentions, len(retained_edges)])
            for edge in retained_edges:
                csv_writer.writerow([edge["from"], edge["to"], edge["weight"] if edge["weight"] > 0 else 0,
                                     -edge["weight"] if edge["weight"] < 0 else 0])
        logger.info(f"Correlation data saved to: {output_path}/correlation_data.tsv")
        # Dump the gold entity label for each mention
        mention_gold_cui_idxs = list(map(lambda x: x['label_idxs'][0], mention_data))
        with open(f'{output_path}/correlation_data_men_labels.json', mode='w') as fh:
            json.dump(mention_gold_cui_idxs, fh)

        # Calculate sample mean and variance of the gold within- and across- cluster edges
        if params["compute_gold_intersection"]:
            within_weights, across_weights = [], []
            for i, edge in enumerate(retained_edges):
                from_label = mention_gold_cui_idxs[edge['from']]
                to_label = mention_gold_cui_idxs[edge['to']]
                if from_label == to_label:
                    within_weights.append(edge['weight'])
                else:
                    across_weights.append(edge['weight'])
            m1, m2 = np.mean(across_weights), np.mean(within_weights)
            std1, std2 = np.std(across_weights), np.std(within_weights)
            intersect_wt = solve_gaussian(m1, m2, std1, std2)
            logger.info(f"Intersection of the within- and across-edge weight distributions of the gold cluster" +
                        f" = {intersect_wt}")

        exit(0)

    # Pickle the graphs
    print(f"Saving top-{knn} candidates")
    with open(cands_path, 'wb') as write_handle:
        pickle.dump(men_cands, write_handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Output final results
    mean_recall_knn = np.mean(men_recall_knn)
    logger.info(f"Result: Final mean recall@{knn} = {mean_recall_knn}")


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()
    args = parser.parse_args()
    print(args)
    main(args.__dict__)
