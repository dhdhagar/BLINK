# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import torch
from torch.utils.data import (DataLoader, SequentialSampler)
import numpy as np
from tqdm import tqdm
import pickle
import faiss
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from special_partition.special_partition import cluster_linking_partition
from collections import defaultdict
import blink.biencoder.data_process_mult as data
import blink.candidate_ranking.utils as utils
from blink.common.params import BlinkParser
from blink.biencoder.biencoder import BiEncoderRanker

from IPython import embed


def embed_and_index(model,
                    token_id_vecs,
                    encoder_type):
    """
    Parameters
    ----------
    model : BiEncoderRanker
        trained biencoder model
    token_id_vecs : ndarray
        list of token id vectors to embed and index
    encoder_type : str
        "context" or "candidate"

    Returns
    -------
    embeds : ndarray
        matrix of embeddings
    index : faiss
        faiss index of the embeddings
    """
    if encoder_type == 'context':
        encoder = model.encode_context
    elif encoder_type == 'candidate':
        encoder = model.encode_candidate
    else:
        raise ValueError("Invalid encoder_type: expected context or candidate")

    # Compute embeddings
    embeds = None
    sampler = SequentialSampler(token_id_vecs)
    dataloader = DataLoader(
        token_id_vecs, sampler=sampler, batch_size=32
    )
    iter_ = tqdm(dataloader, desc="Embedding")
    for step, batch in enumerate(iter_):
        batch_embeds = encoder(batch.cuda())
        embeds = batch_embeds if embeds is None else np.concatenate((embeds, batch_embeds), axis=0)

    # Build index
    d = embeds.shape[1]
    nembeds = embeds.shape[0]
    if nembeds < 10000:  # if the number of embeddings is small, don't approximate
        index = faiss.IndexFlatIP(d)
        index.add(embeds)
    else:
        # number of quantized cells
        nlist = int(math.floor(math.sqrt(nembeds)))
        # number of the quantized cells to probe
        nprobe = int(math.floor(math.sqrt(nlist)))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(
            quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
        )
        index.train(embeds)
        index.add(embeds)
        index.nprobe = nprobe

    # Return embeddings and indexes
    return embeds, index


def get_query_nn(model,
                 knn,
                 embeds,
                 index,
                 q_embed):
    """
    Parameters
    ----------
    model : BiEncoderRanker
        trained biencoder model
    knn : int
        the number of nearest-neighbours to retrieve
    embeds : ndarray
        matrix of embeddings
    index : faiss
        faiss index of the embeddings
    q_dense_embed : ndarray
        2-D array containing the query embedding

    Returns
    -------
    nn_idxs : array
        nearest neighbour indices for the query, sorted in descending order of scores
    scores : array
        similarity scores for each nearest neighbour, sorted in descending order
    """
    # To accomodate the approximate-nature of the knn procedure, retrieve more samples and then filter down
    k = max(16, 2*knn)

    # Find k nearest neighbours
    _, nn_idxs = index.search(q_embed, k)
    nn_idxs = nn_idxs.astype(np.int64).flatten()
    nn_embeds = torch.tensor(list(map(lambda x: embeds[x], nn_idxs))).cuda()

    # Compute query-candidate similarity scores
    scores = torch.flatten(
        torch.mm(torch.tensor(q_embed).cuda(), nn_embeds.T)).cpu()

    # Sort the candidates by descending order of scores
    nn_idxs, scores = zip(
        *sorted(zip(nn_idxs, scores), key=lambda x: -x[1]))

    # Return only the top k neighbours
    return np.array(nn_idxs[:knn]), np.array(scores[:knn])


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
        pred_entity_cuis = [pred_entity['cui']]
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
            men_golden_cuis = men_query['label_cuis']
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

    # Init model
    reranker = BiEncoderRanker(params)
    reranker.model.eval()
    tokenizer = reranker.tokenizer
    model = reranker.model
    device = reranker.device

    knn = params["knn"]
    directed_graph = params["directed_graph"]

    # Load test data
    test_samples = utils.read_dataset("test", params["data_path"])
    if params["filter_unlabeled"]:
        # Filter samples without gold entities
        test_samples = list(filter(lambda sample: len(sample["labels"]) > 0, test_samples))
    logger.info("Read %d test samples." % len(test_samples))

    mention_data, test_dictionary, test_tensor_data = data.process_mention_data(
        test_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params["context_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
        knn=knn
    )

    # Store test dictionary token ids
    test_dict_vecs = torch.tensor(
        list(map(lambda x: x['ids'], test_dictionary)), dtype=torch.long)
    # Store test mention token ids
    test_men_vecs = test_tensor_data[:][0]

    n_entities = len(test_dict_vecs)
    n_mentions = len(test_tensor_data)

    # Values of k to run the evaluation against
    knn_vals = [0] + [2**i for i in range(int(__import__('math').log(knn, 2)) + 1)]
    # Store the maximum evaluation k
    max_knn = knn_vals[-1]

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
                'shape': (n_entities+n_mentions, n_entities+n_mentions)
            }

        # Embed entity dictionary and build indexes
        print("Dictionary: Embedding and building index")
        dict_embeds, dict_index = embed_and_index(
            reranker, test_dict_vecs, 'candidate')

        # Embed mention queries and build indexes
        print("Queries: Embedding and building index")
        men_embeds, men_index = embed_and_index(
            reranker, test_men_vecs, 'context')

        # Find the most similar entity and k-nn mentions for each mention query
        for men_query_idx, men_embed in enumerate(tqdm(men_embeds, total=len(men_embeds), desc="Fetching k-NN")):
            men_embed = np.expand_dims(men_embed, axis=0)

            # Fetch nearest entity candidate
            dict_cand_idx, dict_cand_score = get_query_nn(
                reranker, 1, dict_embeds, dict_index, men_embed)

            # Fetch (k+1) NN mention candidates
            men_cand_idxs, men_cand_scores = get_query_nn(
                reranker, max_knn + 1, men_embeds, men_index, men_embed)
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
