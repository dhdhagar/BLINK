# Copyright (c) Facebook, Inc. and its affiliates.
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
from torch.utils.data import (DataLoader, SequentialSampler)
import numpy as np
from tqdm import tqdm
import pickle
import faiss
from itertools import compress
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from special_partition.special_partition import cluster_linking_partition
from collections import defaultdict
import blink.biencoder.data_process_mult as data_process
import blink.candidate_ranking.utils as utils
from blink.common.params import BlinkParser
from blink.biencoder.biencoder import BiEncoderRanker

from IPython import embed


def partition_graph(graph, n_entities, directed, return_clusters=False, exclude=set(), threshold=None, without_entities=False):
    rows, cols, data, shape = graph['rows'], graph['cols'], graph['data'], graph['shape']

    if not without_entities:
        rows, cols, data = cluster_linking_partition(
            rows,
            cols,
            data,
            n_entities,
            directed,
            exclude=exclude,
            threshold=threshold
        )
    else:
        # Manual filtering that special partition executes
        seen = set()
        duplicated, excluded, thresholded = 0, 0, 0
        _f_row, _f_col, _f_data = [], [], []
        for k in range(len(rows)):
            if (rows[k], cols[k]) in seen:
                duplicated += 1
                continue
            seen.add((rows[k], cols[k]))
            if rows[k] in exclude or cols[k] in exclude:
                excluded += 1
                continue
            if threshold is not None and data[k] < threshold:
                thresholded += 1
                continue
            _f_row.append(rows[k])
            _f_col.append(cols[k])
            _f_data.append(data[k])
        rows, cols, data = list(map(np.array, (_f_row, _f_col, _f_data)))
        if duplicated + excluded + thresholded > 0:
            print(f"""
Dropped edges during pre-processing:
    Duplicates: {duplicated}
    Excluded: {excluded}
    Thresholded: {thresholded}""")

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


def analyzeClusters(clusters, gold_cluster_labels, n_entities, n_mentions, logger):
    logger.info("Analyzing clusters...")

    predicted_cluster_labels = [-1*i for i in range(1, n_mentions+1)]
    n_predicted = 0
    for cluster in clusters.values():
        cluster_label = cluster[0]
        for i in range(len(cluster)):
            men_idx = cluster[i] - n_entities
            if men_idx < 0:
                continue
            predicted_cluster_labels[men_idx] = cluster_label
            n_predicted += 1
    
    debug_no_pred = 0
    for l in predicted_cluster_labels:
        if l < 0:
            debug_no_pred += 1
    assert n_predicted + debug_no_pred == n_mentions

    logger.info(f"{n_predicted} mentions assigned to {len(clusters)} clusters; {debug_no_pred} singelton clusters")

    nmi = normalized_mutual_info_score(gold_cluster_labels, predicted_cluster_labels)
    rand_index = adjusted_rand_score(gold_cluster_labels, predicted_cluster_labels)
    result = (nmi + rand_index) / 2
    logger.info(f"NMI={nmi}, rand_index={rand_index} => average={result}")

    return {'rand_index': rand_index, 'nmi': nmi, 'average': result}

def main(params):
    time_start = time.time()

    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = utils.get_logger(params["output_path"], 'log-discovery')

    embed_data_path = params["embed_data_path"]
    if embed_data_path is None or not os.path.exists(embed_data_path):
        embed_data_path = output_path

    pickle_src_path = params["pickle_src_path"]
    if pickle_src_path is None or not os.path.exists(pickle_src_path):
        pickle_src_path = output_path

    rng = np.random.default_rng(seed=17)
    knn = params["knn"]
    use_types = params["use_types"]
    data_split = params["data_split"] # Default = "test"
    graph_mode = params.get('graph_mode', None)

    logger.info(f"Dataset: {data_split.upper()}")

    # Load evaluation data
    entity_dictionary_loaded = False
    dictionary_pkl_path = os.path.join(pickle_src_path, 'test_dictionary.pickle')
    tensor_data_pkl_path = os.path.join(pickle_src_path, 'test_tensor_data.pickle')
    mention_data_pkl_path = os.path.join(pickle_src_path, 'test_mention_data.pickle')
    print("Loading stored processed entity dictionary...")
    with open(dictionary_pkl_path, 'rb') as read_handle:
        dictionary = pickle.load(read_handle)
    print("Loading stored processed mention data...")
    with open(tensor_data_pkl_path, 'rb') as read_handle:
        tensor_data = pickle.load(read_handle)
    with open(mention_data_pkl_path, 'rb') as read_handle:
        mention_data = pickle.load(read_handle)
    print("Loading embed data...")
    # Check and load stored embedding data
    embed_data_path = os.path.join(embed_data_path, 'embed_data.t7')
    embed_data = torch.load(embed_data_path)
    # Load stored joint graphs
    graph_path = os.path.join(output_path, 'graphs.pickle')
    print("Loading stored joint graphs...")
    with open(graph_path, 'rb') as read_handle:
        joint_graphs = pickle.load(read_handle)

    n_entities = len(dictionary)
    n_mentions = len(mention_data)
    n_labels = 1 # Zeshel and MedMentions have single gold entity mentions

    mention_gold_cui_idxs = list(map(lambda x: x['label_idxs'][n_labels - 1], mention_data))
    ents_in_data = np.unique(mention_gold_cui_idxs)

    if params['drop_all_entities']:
        ent_drop_prop = 1
        n_ents_dropped = len(ents_in_data)
        n_mentions_wo_gold_ents = n_mentions
        logger.info(f"Dropping all {n_ents_dropped} entities found in mention set")
        set_dropped_ent_idxs = set()
    else:
        # Percentage of entities from the mention set to drop
        ent_drop_prop = 0.1
        
        logger.info(f"Dropping {ent_drop_prop*100}% of {len(ents_in_data)} entities found in mention set")

        # Get entity indices to drop
        n_ents_dropped = int(ent_drop_prop*len(ents_in_data))
        dropped_ent_idxs = rng.choice(ents_in_data, size=n_ents_dropped, replace=False)
        set_dropped_ent_idxs = set(dropped_ent_idxs)
        
        n_mentions_wo_gold_ents = sum([1 if x in set_dropped_ent_idxs else 0 for x in mention_gold_cui_idxs])
        logger.info(f"Dropped {n_ents_dropped} entities")
        logger.info(f"=> Mentions without gold entities = {n_mentions_wo_gold_ents}")

        # Load embeddings in order to compute new KNN entities after dropping
        print('Computing new dictionary indexes...')
        
        original_dict_embeds = embed_data['dict_embeds']
        keep_mask = np.ones(len(original_dict_embeds), dtype='bool')
        keep_mask[dropped_ent_idxs] = False
        dict_embeds = original_dict_embeds[keep_mask]

        new_to_old_dict_mapping = []
        for i in range(len(original_dict_embeds)):
            if keep_mask[i]:
                new_to_old_dict_mapping.append(i)

        men_embeds = embed_data['men_embeds']
        if use_types:
            dict_idxs_by_type = data_process.get_idxs_by_type(list(compress(dictionary, keep_mask)))
            dict_indexes = data_process.get_index_from_embeds(dict_embeds, dict_idxs_by_type, force_exact_search=params['force_exact_search'], probe_mult_factor=params['probe_mult_factor'])
            if 'men_idxs_by_type' in embed_data:
                men_idxs_by_type = embed_data['men_idxs_by_type']
            else:
                men_idxs_by_type = data_process.get_idxs_by_type(mention_data)
        else:
            dict_index = data_process.get_index_from_embeds(dict_embeds, force_exact_search=params['force_exact_search'], probe_mult_factor=params['probe_mult_factor'])
        
        # Fetch additional KNN entity to make sure every mention has a linked entity after dropping
        extra_entity_knn = []
        if use_types:
            for men_type in men_idxs_by_type:
                dict_index = dict_indexes[men_type]
                dict_type_idx_mapping = dict_idxs_by_type[men_type]
                q_men_embeds = men_embeds[men_idxs_by_type[men_type]] # np.array(list(map(lambda x: men_embeds[x], men_idxs_by_type[men_type])))
                fetch_k = 1 if isinstance(dict_index, faiss.IndexFlatIP) else 16
                _, nn_idxs = dict_index.search(q_men_embeds, fetch_k)
                for i, men_idx in enumerate(men_idxs_by_type[men_type]):
                    r = n_entities + men_idx
                    q_nn_idxs = dict_type_idx_mapping[nn_idxs[i]]
                    q_nn_embeds = torch.tensor(dict_embeds[q_nn_idxs]).cuda()
                    q_scores = torch.flatten(
                        torch.mm(torch.tensor(q_men_embeds[i:i+1]).cuda(), q_nn_embeds.T)).cpu()
                    c, data = new_to_old_dict_mapping[q_nn_idxs[torch.argmax(q_scores)]], torch.max(q_scores)
                    extra_entity_knn.append((r,c,data))
        else:
            fetch_k = 1 if isinstance(dict_index, faiss.IndexFlatIP) else 16
            _, nn_idxs = dict_index.search(men_embeds, fetch_k)
            for men_idx, men_embed in enumerate(men_embeds):
                r = n_entities + men_idx
                q_nn_idxs = nn_idxs[men_idx]
                q_nn_embeds = torch.tensor(dict_embeds[q_nn_idxs]).cuda()
                q_scores = torch.flatten(
                    torch.mm(torch.tensor(np.expand_dims(men_embed, axis=0)).cuda(), q_nn_embeds.T)).cpu()
                c, data = new_to_old_dict_mapping[q_nn_idxs[torch.argmax(q_scores)]], torch.max(q_scores)
                extra_entity_knn.append((r,c,data))
        
        # Add entities for mentions whose 
        for k in joint_graphs:
            rows, cols, data= [], [], []
            for edge in extra_entity_knn:
                rows.append(edge[0])
                cols.append(edge[1])
                data.append(edge[2])
            joint_graphs[k]['rows'] = np.concatenate((joint_graphs[k]['rows'], rows))
            joint_graphs[k]['cols'] = np.concatenate((joint_graphs[k]['cols'], cols))
            joint_graphs[k]['data'] = np.concatenate((joint_graphs[k]['data'], data))

    results = {
        'data_split': data_split.upper(),
        'n_entities': n_entities,
        'n_mentions': n_mentions,
        'n_entities_dropped': f"{n_ents_dropped} ({ent_drop_prop*100}%)",
        'n_mentions_wo_gold_entities': n_mentions_wo_gold_ents
    }
    if graph_mode is None or graph_mode not in ['directed', 'undirected']:
        graph_mode = ['directed', 'undirected']
    else:
        graph_mode = [graph_mode]

    n_thresholds = params['n_thresholds'] # Default is 10
    exact_threshold = params.get('exact_threshold', None)
    exact_knn = params.get('exact_knn', None)
    
    kmeans = KMeans(n_clusters=n_thresholds, random_state=17)

    # TODO: Baseline? (without dropping entities)

    for mode in graph_mode:
        best_result = -1.
        best_config = None
        for k in joint_graphs:
            if params['drop_all_entities']:
                # Drop all entities from the graph
                rows, cols, data = joint_graphs[k]['rows'], joint_graphs[k]['cols'], joint_graphs[k]['data']
                _f_row, _f_col, _f_data = [], [], []
                for ki in range(len(joint_graphs[k]['rows'])):
                    if joint_graphs[k]['cols'][ki] < n_entities or joint_graphs[k]['rows'][ki] < n_entities:
                        continue
                    _f_row.append(joint_graphs[k]['rows'][ki])
                    _f_col.append(joint_graphs[k]['cols'][ki])
                    _f_data.append(joint_graphs[k]['data'][ki])
                joint_graphs[k]['rows'], joint_graphs[k]['cols'], joint_graphs[k]['data'] = list(map(np.array, (_f_row, _f_col, _f_data)))
            if (exact_knn is None and k > 0 and k <= knn) or (exact_knn is not None and k == exact_knn):
                if exact_threshold is not None:
                    thresholds = np.array([0, exact_threshold])
                else:
                    thresholds = np.sort(np.concatenate(([0], kmeans.fit(joint_graphs[k]['data'].reshape(-1,1)).cluster_centers_.flatten())))
                for thresh in thresholds:
                    print("\nPartitioning...")
                    logger.info(f"{mode.upper()}, k={k}, threshold={thresh}")
                    # Partition graph based on cluster-linking constraints
                    partitioned_graph, clusters = partition_graph(
                        joint_graphs[k], n_entities, mode == 'directed', return_clusters=True, exclude=set_dropped_ent_idxs, threshold=thresh, without_entities=params['drop_all_entities'])
                    # Analyze cluster against gold clusters
                    result = analyzeClusters(clusters, mention_gold_cui_idxs, n_entities, n_mentions, logger)
                    results[f'({mode}, {k}, {thresh})'] = result
                    if thresh != 0 and result['average'] > best_result:
                        best_result = result['average']
                        best_config = (mode, k, thresh)
        results[f'best_{mode}_config'] = best_config
        results[f'best_{mode}_result'] = best_result
    
    # Store results
    output_file_name = os.path.join(
        output_path, f"{data_split}_eval_discovery_{__import__('calendar').timegm(__import__('time').gmtime())}.json")

    with open(output_file_name, 'w') as f:
        json.dump(results, f, indent=2)
        print(f"\nAnalysis saved at: {output_file_name}")
    execution_time = (time.time() - time_start) / 60
    logger.info(f"\nTotal time taken: {execution_time} minutes\n")


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()
    args = parser.parse_args()
    print(args)
    main(args.__dict__)
