from collections import defaultdict
from copy import deepcopy
import logging
import numpy as np
import pickle
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.csgraph import (minimum_spanning_tree,
                                  connected_components,
                                  breadth_first_tree)
from sklearn.cluster import KMeans
from sklearn.metrics import (fowlkes_mallows_score,
                             adjusted_rand_score)
from tqdm import tqdm

from blink.joint.joint_eval.special_partition import special_partition

from IPython import embed


logger = logging.getLogger(__name__)


def eval_wdoc(args,
              example_dir,
              metadata,
              sub_trainer,
              save_fname=None):
    assert save_fname != None

    logger.info('Building within doc sparse graphs...')
    doc_level_graphs = []
    per_doc_coref_clusters = []
    for doc_clusters in tqdm(metadata.wdoc_clusters.values(), disable=(get_rank() != 0)):
        per_doc_coref_clusters.append(
                [[x for x in v if x != k] for k, v in doc_clusters.items()]
        )
        doc_mentions = np.asarray([x for k, v in doc_clusters.items()
                                        for x in v if x != k])
        doc_mentions = np.sort(doc_mentions)
        doc_level_graphs.append(
            build_sparse_affinity_graph(
                args,
                doc_mentions,
                example_dir,
                metadata,
                None,
                sub_trainer,
                build_coref_graph=True,
                build_linking_graph=True
            )
        )

    logger.info('Done.')

    # don't need other processes at this point
    if get_rank() != 0:
        synchronize()
        return

    # build everything needed to compute metrics and compute them!
    coref_graphs, linking_graphs = [], []
    for coref_graph, linking_graph in doc_level_graphs:
        coref_graphs.append(coref_graph)
        linking_graphs.append(linking_graph)

    logger.info('Computing coref metrics...')
    coref_metrics = compute_coref_metrics(
            per_doc_coref_clusters, coref_graphs, args.eval_coref_threshold
    )
    logger.info('Done.')

    logger.info('Computing linking metrics...')
    linking_metrics, slim_linking_graph = compute_linking_metrics(
            metadata, linking_graphs
    )
    logger.info('Done.')

    logger.info('Computing joint metrics...')
    slim_coref_graph = _get_global_maximum_spanning_tree(coref_graphs)
    joint_metrics = compute_joint_metrics(metadata,
                                          [slim_coref_graph, slim_linking_graph])
    logger.info('Done.')

    metrics = {
        'coref_fmi' : coref_metrics['fmi'],
        'coref_rand_index' : coref_metrics['rand_index'],
        'coref_threshold' : coref_metrics['threshold'],
        'vanilla_recall' : linking_metrics['vanilla_recall'],
        'vanilla_accuracy' : linking_metrics['vanilla_accuracy'],
        'joint_accuracy' : joint_metrics['joint_accuracy'],
        'joint_cc_recall' : joint_metrics['joint_cc_recall']
    }

    # save all of the predictions for later analysis
    save_data = {}
    save_data.update(coref_metrics)
    save_data.update(linking_metrics)
    save_data.update(joint_metrics)
    save_data.update({'metadata': metadata})

    with open(save_fname, 'wb') as f:
        pickle.dump(save_data, f)

    synchronize()
    return metrics


def compute_coref_metrics(
    gold_coref_clusters,
    coref_graph,
    coref_threshold=None
):
    global_maximum_spanning_tree = _get_global_maximum_spanning_tree(
            [coref_graph]
    )

    # compute metrics and choose threshold is one isn't specified
    if coref_threshold is None:
        logger.info('Generating candidate thresholds...')
        _edge_weights = global_maximum_spanning_tree.data.reshape(-1, 1)
        _num_thresholds = 1000
        if _edge_weights.shape[0] < _num_thresholds:
            candidate_thresholds = _edge_weights.reshape(-1,).tolist()
        else:
            kmeans = KMeans(n_clusters=_num_thresholds, random_state=0)
            kmeans.fit(global_maximum_spanning_tree.data.reshape(-1, 1))
            candidate_thresholds = kmeans.cluster_centers_.reshape(-1,).tolist()
        logger.info('Done.')

        logger.info('Choosing threshold...')
        threshold_results = []
        for _threshold in tqdm(candidate_thresholds):
            _metrics = _compute_coref_metrics_threshold(
                    gold_coref_clusters,
                    global_maximum_spanning_tree,
                    _threshold
            )
            threshold_results.append((_threshold, _metrics))
        logger.info('Done.')
        max_threshold_results = max(threshold_results,
                                    key=lambda x : x[1]['rand_index'])

        coref_results = max_threshold_results[1]
        coref_results['threshold'] = max_threshold_results[0]
    else:
        coref_results = _compute_coref_metrics_threshold(
                    gold_coref_clusters,
                    global_maximum_spanning_tree,
                    coref_threshold
        )
        coref_results['threshold'] = coref_threshold

    coref_results['global_coref_mst'] = global_maximum_spanning_tree
    coref_results['gold_coref_clusters'] = gold_coref_clusters
        
    return coref_results


def compute_linking_metrics(linking_graph, gold_linking_map, seen_uids=None):
    global_graph = _merge_sparse_graphs([linking_graph])

    # compute recall
    _row = global_graph.row
    _col = global_graph.col
    mention2cand = defaultdict(list)
    for eidx, midx in zip(_row, _col):
        assert eidx < midx
        mention2cand[midx].append(eidx)

    midxs = list(gold_linking_map.keys())
    recall_hits = 0
    recall_total = len(midxs)
    no_candidates = []
    for midx in midxs:
        cands = mention2cand.get(midx, [])
        if len(cands) == 0:
            no_candidates.append(midx)
            continue
        if gold_linking_map[midx] in cands:
            recall_hits += 1

    midxs = np.asarray(midxs)

    # build slim global linking graph for joint linking inference
    global_graph = global_graph.tocsc()
    def _get_slim_links(midx):
        col_entries = global_graph.getcol(midx).tocoo()
        if col_entries.nnz == 0:
            return (0, -np.inf)
        return max(zip(col_entries.row, col_entries.data), key=lambda x : x[1])
    v_max = np.vectorize(_get_slim_links)
    pred_eidxs, max_affinities = v_max(midxs)
    slim_global_graph = coo_matrix((max_affinities, (pred_eidxs, midxs)),
                                   shape=global_graph.shape)

    # compute linking accuracy
    missed_vanilla_midxs = []
    linking_hits, linking_total = 0, 0
    seen_hits, unseen_hits, seen_misses, unseen_misses = 0, 0, 0, 0
    pred_midx2eidx = {m : e for m, e in zip(midxs, pred_eidxs)}
    for midx, true_eidx in gold_linking_map.items():
        if true_eidx == pred_midx2eidx.get(midx, -1):
            linking_hits += 1
            if seen_uids is not None and true_eidx in seen_uids:
                seen_hits += 1
            elif seen_uids is not None and true_eidx not in seen_uids:
                unseen_hits += 1
        else:
            if true_eidx in mention2cand[midx]:
                missed_vanilla_midxs.append(midx)
            if seen_uids is not None and true_eidx in seen_uids:
                seen_misses += 1
            elif seen_uids is not None and true_eidx not in seen_uids:
                unseen_misses += 1
        linking_total += 1

    results_dict = {
            'vanilla_recall' : recall_hits / recall_total,
            'vanilla_accuracy' : linking_hits / linking_total,
            'num_no_candidates' : len(no_candidates),
            'vanilla_pred_midx2eidx' : {m : e for m, e in zip(midxs, pred_eidxs)},
            'vanilla_slim_graph' : slim_global_graph,
    }
    if seen_uids is not None:
        results_dict.update({
            'seen_accuracy' : seen_hits / (seen_hits + seen_misses),
            'unseen_accuracy' : unseen_hits / (unseen_hits + unseen_misses)
        })

    return results_dict, slim_global_graph


def compute_joint_metrics(joint_graphs, gold_linking_map, num_entities):
    # get global joint graph
    global_joint_graph = _merge_sparse_graphs(joint_graphs)

    # compute recall at this stage
    _, cc_labels = connected_components(
            csgraph=global_joint_graph,
            directed=False,
            return_labels=True
    )
    cc_recall_hits, cc_recall_total = 0, 0
    for midx, true_eidx in gold_linking_map.items():
        if cc_labels[midx] == cc_labels[true_eidx]:
            cc_recall_hits += 1
        cc_recall_total += 1
    cc_recall = cc_recall_hits / cc_recall_total

    # reorder the data for the requirements of the `special_partition` function
    _row = np.concatenate((global_joint_graph.row, global_joint_graph.col))
    _col = np.concatenate((global_joint_graph.col, global_joint_graph.row))
    _data = np.concatenate((global_joint_graph.data, global_joint_graph.data))
    tuples = zip(_row, _col, _data)
    tuples = sorted(tuples, key=lambda x : (x[1], -x[0])) # sorted this way for nice DFS
    special_row, special_col, special_data = zip(*tuples)
    special_row = np.asarray(special_row, dtype=np.int)
    special_col = np.asarray(special_col, dtype=np.int)
    special_data = np.asarray(special_data)

    # reconstruct the global joint graph shape
    global_joint_graph = coo_matrix(
            (special_data, (special_row, special_col)),
            shape=global_joint_graph.shape
    )

    # create siamese indices for easy lookups during partitioning
    edge_indices = {e : i for i, e in enumerate(zip(special_row, special_col))}
    siamese_indices = [edge_indices[(c, r)]
                            for r, c in zip(special_row, special_col)]
    siamese_indices = np.asarray(siamese_indices)

    # order the edges in ascending order according to affinities
    ordered_edge_indices = np.argsort(special_data)

    # partition the graph using the `keep_edge_mask`
    keep_edge_mask = special_partition(
            special_row,
            special_col,
            ordered_edge_indices,
            siamese_indices,
            num_entities
    )

    # build the partitioned graph
    partitioned_joint_graph = coo_matrix(
            (special_data[keep_edge_mask],
             (special_row[keep_edge_mask], special_col[keep_edge_mask])),
            shape=global_joint_graph.shape
    )

    # infer the linking decisions from clusters (connected compoents) of
    # the partitioned joint mention and entity graph 
    _, labels = connected_components(
            csgraph=partitioned_joint_graph,
            directed=False,
            return_labels=True
    )
    unique_labels, cc_sizes = np.unique(labels, return_counts=True)
    components = defaultdict(list)
    filtered_labels = unique_labels[cc_sizes > 1]
    for idx, cc_label in enumerate(labels):
        if cc_label in filtered_labels:
            components[cc_label].append(idx)
    pred_midx2eidx = {}
    for cluster_nodes in components.values():
        eidxs = [x for x in cluster_nodes if x < num_entities]
        midxs = [x for x in cluster_nodes if x >= num_entities]
        assert len(eidxs) == 1
        assert len(midxs) >= 1
        eidx = eidxs[0]
        for midx in midxs:
            pred_midx2eidx[midx] = eidx

    joint_hits, joint_total = 0, 0
    for midx, true_eidx in gold_linking_map.items():
        if pred_midx2eidx.get(midx, -1) == true_eidx:
            joint_hits += 1
        joint_total += 1

    return {'joint_accuracy' : joint_hits / joint_total,
            'joint_pred_midx2eidx': pred_midx2eidx,
            'joint_cc_recall': cc_recall,
            'joint_slim_graph': global_joint_graph,
            'joint_keep_edge_mask': keep_edge_mask}


def _compute_coref_metrics_threshold(gold_clustering, mst, threshold):
    # get the true_labels 
    true_labels = [(i, x) for i, l in enumerate(gold_clustering) for x in l]
    true_labels = sorted(true_labels, key=lambda x : x[1])
    true_labels, true_midxs = zip(*true_labels)

    # prune mst (make sure to deepcopy!!!!)
    pruned_mst = deepcopy(mst)
    pruned_mask = pruned_mst.data > threshold
    pruned_mst.row = pruned_mst.row[pruned_mask]
    pruned_mst.col = pruned_mst.col[pruned_mask]
    pruned_mst.data = pruned_mst.data[pruned_mask]

    # get connected components to get clusters of `pruned_mst`
    n_components, labels = connected_components(
            csgraph=pruned_mst, directed=False, return_labels=True
    )
    pred_midxs = np.arange(labels.size)
    label_mask = np.isin(pred_midxs, true_midxs)
    pred_labels = zip(labels[label_mask], pred_midxs[label_mask])
    pred_labels = sorted(pred_labels, key=lambda x : x[1])
    pred_labels, _ = zip(*pred_labels)

    return {'fmi' : fowlkes_mallows_score(true_labels, pred_labels),
            'rand_index' : adjusted_rand_score(true_labels, pred_labels),
            'pred_labels' : pred_labels,
            'true_labels' : true_labels,
            'midxs' : true_midxs}


def _merge_sparse_graphs(graphs):
    global_graph_row = np.concatenate([graph.row for graph in graphs])
    global_graph_col = np.concatenate([graph.col for graph in graphs])
    global_graph_data = np.concatenate([graph.data for graph in graphs])
    global_graph = coo_matrix(
            (global_graph_data, (global_graph_row, global_graph_col)),
            shape=graphs[0].shape
    )
    return global_graph


def _get_global_maximum_spanning_tree(sparse_graph_list):
    # get MAXIMUM spanning trees by flipping affinities and computing
    # minimum spanning trees using these flipped affinities then flippling
    # the MST weights back
    for g in sparse_graph_list:
        g.data *= -1.0

    msts = [minimum_spanning_tree(g).tocoo() for g in sparse_graph_list]

    # NOTE: !!!! have to flip back in memory stuff !!!!
    for g in sparse_graph_list:
        g.data *= -1.0

    # make `msts` into MAXIMUM spanning trees 
    for mst in msts:
        mst.data *= -1.0

    # merge per doc things to global
    global_maximum_spanning_tree = _merge_sparse_graphs(msts)

    return global_maximum_spanning_tree
