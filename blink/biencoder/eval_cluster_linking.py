# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import json
import math
import time
import torch
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np
from tqdm import tqdm
import pickle
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components  # CC2
from special_partition.special_partition import (
    cluster_linking_partition,
)  # DD4 Implements MST + inference procedure

from collections import defaultdict
import blink.biencoder.data_process_mult as data_process
import blink.candidate_ranking.utils as utils
from blink.common.params import BlinkParser
from blink.biencoder.biencoder import BiEncoderRanker

from IPython import embed


def get_query_nn(
    knn, embeds, index, q_embed, searchK=None, gold_idxs=None, type_idx_mapping=None
):
    """
    Retrieves the top k neighbours of the query embedding (We know it's for mentions because of knn)

    Parameters
    ----------
    knn : int
        the number of nearest-neighbours to return
    embeds : ndarray of dim (number_mentions, H: embedding_size)
        matrix of embeddings
    index : faiss
        faiss index of the embeddings
    q_embed : ndarray of dim (1, H)
        2-D array containing the query embedding
    searchK: int
        optional parameter, the exact number of nearest-neighbours to retrieve and score
        (Used for approximate search)
    gold_idxs : array
        optional parameter, list of golden cui indexes (= the correct mentions for the query).
    type_idx_mapping : array
        optional parameter, list mapping type-specific indexes to the indexes of the full dictionary

    Returns
    -------
    nn_idxs : array
        indices of the nearest neighbour, sorted in descending order of scores
    scores : array
        similarity scores of the nearest neighbors, sorted in descending order
    """
    # To accomodate the approximate-nature of the knn procedure, retrieve more samples and then filter down
    k = (
        searchK if searchK is not None else max(16, 2 * knn)
    )  # 'is not None' not necessary

    # Find k nearest neighbours of the query
    _, nn_idxs = index.search(q_embed, k)
    nn_idxs = nn_idxs.astype(
        np.int64
    ).flatten()  # 2D array of dim = (1, k) --> 1D array of dim = (k,)
    if type_idx_mapping is not None:
        nn_idxs = type_idx_mapping[nn_idxs]
    nn_embeds = torch.tensor(embeds[nn_idxs]).cuda()

    # Compute query-candidate similarity scores
    scores = torch.flatten(torch.mm(torch.tensor(q_embed).cuda(), nn_embeds.T)).cpu()

    # Sort the candidates by descending order of scores
    nn_idxs, scores = zip(*sorted(zip(nn_idxs, scores), key=lambda x: -x[1]))  # CC1 DD1

    if gold_idxs is not None:
        # Calculate the knn index at which the gold cui is found (-1 if not found)
        # DD2 Only find one even though several can exist
        for topk, i in enumerate(nn_idxs):
            if i in gold_idxs:
                break
            topk = -1
        # Return only the top k neighbours, and the recall index
        return (
            np.array(nn_idxs[:knn], dtype=np.int64),
            np.array(scores[:knn]),
            topk,
        )  # DD3

    # Return only the top k neighbours
    return np.array(nn_idxs[:knn], dtype=np.int64), np.array(scores[:knn])


def partition_graph(
    graph, n_entities, directed, return_clusters=False, exclude=set(), threshold=None
):
    """
    Convert the initial graph constructed in train_biencoder_mst.evaluate(.) into the final graph resulting from the inference procedure

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
    rows, cols, data, shape = (
        graph["rows"],
        graph["cols"],
        graph["data"],
        graph["shape"],
    )

    rows, cols, data = cluster_linking_partition(  # DD4
        rows, cols, data, n_entities, directed, exclude=exclude, threshold=threshold
    )
    # Construct the partitioned graph
    partitioned_graph = coo_matrix((data, (rows, cols)), shape=shape)

    if return_clusters:
        # Get an array of the graph with each index marked with the component label that it is connected to
        _, cc_labels = connected_components(  # CC2
            csgraph=partitioned_graph, directed=directed, return_labels=True
        )
        # Store clusters of indices marked with labels with at least 2 connected components
        unique_cc_labels, cc_sizes = np.unique(cc_labels, return_counts=True)  # CC3
        filtered_labels = unique_cc_labels[cc_sizes >= 2]
        clusters = defaultdict(list)
        for i, cc_label in enumerate(cc_labels):
            if cc_label in filtered_labels:
                clusters[cc_label].append(i)
        return partitioned_graph, clusters

    return partitioned_graph


def analyzeClusters(clusters, dictionary, queries, knn, n_train_mentions=0):
    """
    Evaluates the accuracy of entity-mention clustering

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
        "n_entities": n_entities,
        "n_mentions": n_mentions,
        "knn_mentions": knn,
        "accuracy": 0,
        "failure": [],
        "success": [],
    }
    (
        _debug_n_mens_evaluated,
        _debug_clusters_wo_entities,
        _debug_clusters_w_mult_entities,
    ) = (0, 0, 0)

    print("Analyzing clusters...")
    for cluster in clusters.values():  # CC4 Scans all the values in clusters
        "Evaluate entity"
        # DD5 The lowest value in the cluster should always be the entity
        pred_entity_idx = cluster[0]
        # Track the graph index of the entity in the cluster
        pred_entity_idxs = [pred_entity_idx]
        if pred_entity_idx >= n_entities:
            # If the first element is a mention, then the cluster does not have an entity
            _debug_clusters_wo_entities += 1
            continue  # DD6 skip to the next cluster (for cluster in clusters.values)
        pred_entity = dictionary[pred_entity_idx]
        pred_entity_cuis = [
            str(pred_entity["cui"])
        ]  # Retrieve the cui of the entity indice
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
                pred_entity_cuis += list(
                    set([dictionary[cluster[i]]["cui"]]) - set(pred_entity_cuis)
                )
                continue
            men_idx -= n_train_mentions  # what's n_train_mentions ?
            if men_idx < 0:
                # Query is from train set
                continue
            " For each mentions in the cluster, we compare its mention_gold_cui to the predicted_cui"
            _debug_n_mens_evaluated += 1
            men_query = queries[men_idx]
            men_golden_cuis = list(
                map(str, men_query["label_cuis"])
            )  # One mention can can be associated with several CUIs in certain contexts
            report_obj = {
                "mention_id": men_query["mention_id"],
                "mention_name": men_query["mention_name"],
                "mention_gold_cui": "|".join(men_golden_cuis),  # CC5
                "mention_gold_cui_name": "|".join(
                    [
                        dictionary[i]["title"]
                        for i in men_query["label_idxs"][: men_query["n_labels"]]
                    ]
                ),
                "predicted_name": "|".join(
                    [d["title"] for d in [dictionary[i] for i in pred_entity_idxs]]
                ),
                "predicted_cui": "|".join(pred_entity_cuis),
            }
            # Correct prediction = if there is at least one common element between the predicted entity CUIs (pred_entity_cuis) and the gold standard entity CUIs (men_golden_cuis)
            if not set(pred_entity_cuis).isdisjoint(men_golden_cuis):
                results["accuracy"] += 1
                results["success"].append(report_obj)
            # Incorrect prediction
            else:
                results["failure"].append(report_obj)
    results["accuracy"] = (
        f"{results['accuracy'] / float(_debug_n_mens_evaluated) * 100} %"
    )
    print(f"Accuracy = {results['accuracy']}")
    # print('1st element of results["failure"] :', results["failure"][0])

    # Run sanity checks
    assert n_mentions == _debug_n_mens_evaluated
    assert _debug_clusters_wo_entities == 0
    assert _debug_clusters_w_mult_entities == 0

    return results


def filter_by_context_doc_id(mention_idxs, doc_id, doc_id_list, return_numpy=False):
    """
    Description
    -----------
    Filters and returns only mention indices that belong to a specific document identified by the doc_id.
    Ensures that the analysis are constrained within the context of that particular document.

    Parameters
    ----------
    - mention_idxs : ndarray(int) of dim = (number of mentions)
    Represents the indices of mentions
    - doc_id : int
    Indice of the target document
    - doc_id_list : ndarray(int) of dim = (number of mentions)
    Array of integers, where each element is a document ID associated with the corresponding mention in mention_idxs.
    The length of doc_id_list should match the total number of mentions referenced in mention_idxs.
    - return_numpy : bool
    A flag indicating whether to return the filtered list of mention indices as a NumPy array.
    If True, the function returns a NumPy array; otherwise, it returns a list
    -------
    Outputs:
    - mask : ndarray(bool) of dim = (number of mentions)
    Mask indicating where each mention's document ID (from "doc_id_list") matches the target "doc_id"
    - mention_idxs :
    Only contains mention indices that belong to the target document (=doc_id).
    """
    mask = [doc_id_list[i] == doc_id for i in mention_idxs]
    if isinstance(mention_idxs, list):
        mention_idxs = np.array(mention_idxs)
    mention_idxs = mention_idxs[mask]
    if not return_numpy:
        mention_idxs = list(mention_idxs)
    return mention_idxs, mask


def read_data(split, params, logger):
    """
    Description
    -----------
    Loads dataset samples from a specified path
    Optionally filters out samples without labels
    Checks if the dataset supports multiple labels per sample
    (has_mult_labels : bool)

    Parameters
    ----------
    split : str
        Indicates the portion of the dataset to load ("train", "test", "valid"), used by utils.read_dataset to determine which data to read.
    params : dict(str)
        Contains configuration options
    logger :
        An object used for logging messages about the process, such as the number of samples read.
    """
    samples = utils.read_dataset(split, params["data_path"])
    # Check if dataset has multiple ground-truth labels
    has_mult_labels = "labels" in samples[0].keys()
    if params["filter_unlabeled"]:
        # Filter samples without gold entities
        samples = list(
            filter(
                lambda sample: (
                    (len(sample["labels"]) > 0)
                    if has_mult_labels
                    else (sample["label"] is not None)
                ),
                samples,
            )
        )
    logger.info(f"Read {len(samples)} {split} samples.")
    return samples, has_mult_labels
