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
import numpy as np
from tqdm import trange
import pickle
import higra as hg
from sklearn.preprocessing import normalize

import blink.biencoder.data_process_mult as data_process
import blink.candidate_ranking.utils as utils
from blink.common.params import BlinkParser
from blink.biencoder.biencoder import BiEncoderRanker

from IPython import embed


def get_hac_tree(graph, weights, linkage="single"):
    fn_map = {
        'single': hg.binary_partition_tree_single_linkage,
        'complete': hg.binary_partition_tree_complete_linkage,
        'average': hg.binary_partition_tree_average_linkage
    }
    tree, altitudes = fn_map[linkage](graph, weights)
    return tree


def main(params):
    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = utils.get_logger(params["output_path"], 'log')

    pickle_src_path = params["pickle_src_path"]
    if pickle_src_path is None or not os.path.exists(pickle_src_path):
        pickle_src_path = output_path

    embed_data_path = params["embed_data_path"]
    if embed_data_path is None or not os.path.exists(embed_data_path):
        embed_data_path = output_path

    # Init model
    reranker = BiEncoderRanker(params)
    reranker.model.eval()
    tokenizer = reranker.tokenizer
    n_gpu = reranker.n_gpu

    knn = params["knn"]  # Use as the max-knn value for the graph construction
    use_types = params["use_types"]
    # within_doc = params["within_doc"]
    data_split = params["data_split"]  # Default = "test"

    # Load test data
    entity_dictionary_loaded = False
    test_dictionary_pkl_path = os.path.join(pickle_src_path, 'test_dictionary.pickle')
    test_tensor_data_pkl_path = os.path.join(pickle_src_path, 'test_tensor_data.pickle')
    test_mention_data_pkl_path = os.path.join(pickle_src_path, 'test_mention_data.pickle')
    # if params['transductive']:
    #     train_tensor_data_pkl_path = os.path.join(pickle_src_path, 'train_tensor_data.pickle')
    #     train_mention_data_pkl_path = os.path.join(pickle_src_path, 'train_mention_data.pickle')
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
    
    # Reducing the entity dictionary to only the ground truth of the mention queries
    # Combining the entities and mentions into one structure for joint embedding and indexing
    new_ents = {}
    new_ents_arr = []
    men_labels = []
    for men in mention_data:
        ent = men['label_idxs'][0]
        if ent not in new_ents:
            new_ents[ent] = len(new_ents_arr)
            new_ents_arr.append(ent)
        men_labels.append(new_ents[ent])
    ent_labels = [i for i in range(len(new_ents_arr))]
    new_ent_vecs = torch.tensor(list(map(lambda x: test_dictionary[x]['ids'], new_ents_arr)))
    new_ent_types = list(map(lambda x: {"type": test_dictionary[x]['type']}, new_ents_arr))
    test_men_vecs = test_tensor_data[:][0]

    n_mentions = len(test_tensor_data)
    n_entities = len(new_ent_vecs)
    n_embeds = n_mentions + n_entities
    leaf_labels = np.array(ent_labels + men_labels, dtype=int)
    all_vecs = torch.cat((new_ent_vecs, test_men_vecs))
    all_types = new_ent_types + mention_data  # Array of dicts containing key "type" for selected ents and all mentions

    # Values of k to run the evaluation against
    knn_vals = [25 * 2 ** i for i in range(int(math.log(knn / 25, 2)) + 1)] if params["exact_knn"] is None else [
        params["exact_knn"]]
    # Store the maximum evaluation k
    max_knn = knn_vals[-1]

    time_start = time.time()

    # Check if graphs are already built
    graph_path = os.path.join(output_path, 'graphs.pickle')
    if os.path.isfile(graph_path):
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
                'shape': (n_embeds, n_embeds)
            }

        # Check and load stored embedding data
        embed_data_path = os.path.join(embed_data_path, 'embed_data.t7')
        embed_data = None
        if os.path.isfile(embed_data_path):
            embed_data = torch.load(embed_data_path)
        if use_types:
            if embed_data is not None:
                logger.info('Loading stored embeddings')
                embeds = embed_data['embeds']
                if 'idxs_by_type' in embed_data:
                    idxs_by_type = embed_data['idxs_by_type']
                else:
                    idxs_by_type = data_process.get_idxs_by_type(all_types)
            else:
                logger.info("Embedding data")
                dict_embeds = data_process.embed_and_index(reranker, all_vecs[:n_entities], encoder_type='candidate',
                                                           only_embed=True, n_gpu=n_gpu,
                                                           batch_size=params['embed_batch_size'])
                men_embeds = data_process.embed_and_index(reranker, all_vecs[n_entities:], encoder_type='context',
                                                          only_embed=True, n_gpu=n_gpu,
                                                          batch_size=params['embed_batch_size'])
                embeds = np.concatenate((dict_embeds, men_embeds), axis=0)
                idxs_by_type = data_process.get_idxs_by_type(all_types)
            search_indexes = data_process.get_index_from_embeds(embeds, corpus_idxs=idxs_by_type, force_exact_search=True)
        else:
            if embed_data is not None:
                logger.info('Loading stored embeddings')
                embeds = embed_data['embeds']
            else:
                logger.info("Embedding data")
                dict_embeds = data_process.embed_and_index(reranker, all_vecs[:n_entities], encoder_type='candidate',
                                                           only_embed=True, n_gpu=n_gpu,
                                                           batch_size=params['embed_batch_size'])
                men_embeds = data_process.embed_and_index(reranker, all_vecs[n_entities:], encoder_type='context',
                                                          only_embed=True, n_gpu=n_gpu,
                                                          batch_size=params['embed_batch_size'])
                embeds = np.concatenate((dict_embeds, men_embeds), axis=0)
            search_index = data_process.get_index_from_embeds(embeds, force_exact_search=True)
        # Save computed embedding data if not loaded from disk
        if embed_data is None:
            embed_data = {}
            embed_data['embeds'] = embeds
            if use_types:
                embed_data['idxs_by_type'] = idxs_by_type
            # NOTE: Cannot pickle faiss index because it is a SwigPyObject
            torch.save(embed_data, embed_data_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

        # Build faiss search index
        if params["normalize_embeds"]:
            embeds = normalize(embed, axis=0)
        logger.info("Building KNN index...")
        if use_types:
            search_indexes = data_process.get_index_from_embeds(embeds, corpus_idxs=idxs_by_type,
                                                                force_exact_search=True)
        else:
            search_index = data_process.get_index_from_embeds(embeds, force_exact_search=True)

        logger.info("Starting KNN search...")
        if not use_types:
            faiss_dists, faiss_idxs = search_index.search(embeds, max_knn+1)
        else:
            query_len = n_embeds
            faiss_idxs = np.zeros((query_len, max_knn+1))
            faiss_dists = np.zeros((query_len, max_knn+1), dtype=float)
            for entity_type in search_indexes:
                embeds_by_type = embeds[idxs_by_type[entity_type]]
                nn_dists_by_type, nn_idxs_by_type = search_indexes[entity_type].search(embeds_by_type, max_knn+1)
                for i, idx in enumerate(idxs_by_type[entity_type]):
                    faiss_idxs[idx] = nn_idxs_by_type[i]
                    faiss_dists[idx] = nn_dists_by_type[i]
        logger.info("Search finished")

        logger.info('Building graphs')
        # Find the most similar nodes for each mention and node in the set (minus self)
        for idx in trange(n_embeds):
            # Compute adjacent node edge weight
            if idx != 0:
                adj_idx = idx - 1
                adj_data = embeds[adj_idx] @ embeds[idx]
            nn_idxs = faiss_idxs[idx]
            nn_scores = faiss_dists[idx]
            # Filter candidates to remove mention query and keep only the top k candidates
            filter_mask = nn_idxs != idx
            nn_idxs, nn_scores = nn_idxs[filter_mask][:max_knn], nn_scores[filter_mask][:max_knn]
            # Add edges to the graphs
            for k in joint_graphs:
                # Add edge to adjacent node to force the graph to be connected
                if idx != 0:
                    joint_graph['rows'] = np.append(
                        joint_graph['rows'], adj_idx)
                    joint_graph['cols'] = np.append(
                        joint_graph['cols'], idx)
                    joint_graph['data'] = np.append(
                        joint_graph['data'], adj_data)
                joint_graph = joint_graphs[k]
                # Add mention-mention edges
                joint_graph['rows'] = np.append(
                    joint_graph['rows'], [idx] * k)
                joint_graph['cols'] = np.append(
                    joint_graph['cols'], nn_idxs[:k])
                joint_graph['data'] = np.append(
                    joint_graph['data'], nn_scores[:k])

        knn_fetch_time = time.time() - time_start
        # Pickle the graphs
        print("Saving joint graphs...")
        with open(graph_path, 'wb') as write_handle:
            pickle.dump(joint_graphs, write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        if params['only_embed_and_build']:
            logger.info(f"Saved embedding data at: {embed_data_path}")
            logger.info(f"Saved graphs at: {graph_path}")
            exit()

    results = {
        'n_leaves': n_embeds,
        'n_entities': n_entities,
        'n_mentions': n_mentions
    }

    graph_processing_time = time.time()
    n_graphs_processed = 0.
    linkage_fns = ["single", "complete", "average"]  # Different HAC linkage functions to run the analyses over

    for fn in linkage_fns:
        purities = []
        fn_result = {}
        for k in joint_graphs:
            graph = hg.UndirectedGraph(n_embeds)
            graph.add_edges(joint_graphs[k]['rows'], joint_graphs[k]['cols'])
            weights = -joint_graphs[k]['data']  # Since Higra expects weights as distances, not similarity
            tree = get_hac_tree(graph, weights, linkage=fn)
            purity = hg.dendrogram_purity(tree, leaf_labels)
            fn_result[f"purity@{k}nn"] = purity
            purities.append(purity)
            n_graphs_processed += 1
        fn_result["average"] = round(np.mean(purities), 2)
        results[fn] = fn_result

    avg_graph_processing_time = (time.time() - graph_processing_time) / n_graphs_processed
    avg_per_graph_time = (knn_fetch_time + avg_graph_processing_time) / 60
    execution_time = (time.time() - time_start) / 60

    # Store results
    output_file_name = os.path.join(
        output_path, f"results_{__import__('calendar').timegm(__import__('time').gmtime())}")

    logger.info(f"Results: \n {results}")
    with open(f'{output_file_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
        print(f"\nResults saved at: {output_file_name}.json")
    
    logger.info("\nThe avg. per graph evaluation time is {} minutes\n".format(avg_per_graph_time))
    logger.info("\nThe total evaluation took {} minutes\n".format(execution_time))



if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()
    args = parser.parse_args()
    print(args)
    main(args.__dict__)
