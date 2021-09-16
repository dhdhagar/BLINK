# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import defaultdict
import json
import logging
import numpy as np
import torch
from tqdm import tqdm

from blink.biencoder.zeshel_utils import WORLDS, Stats
import blink.candidate_ranking.utils as utils
from blink.index.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer

from IPython import embed


def get_knn(
    index_vecs,
    query_vecs,
    top_k,
    logger,
    hnsw=False,
    index_buffer=50000
):
    vector_size = index_vecs.size(1)
    if hnsw:
        logger.info("Using HNSW index in FAISS")
        index = DenseHNSWFlatIndexer(vector_size, index_buffer)
    else:
        logger.info("Using Flat index in FAISS")
        index = DenseFlatIndexer(vector_size, index_buffer)

    index.index_data(index_vecs.numpy())
    _, local_idxs = index.search_knn(query_vecs.numpy(), top_k)

    return torch.tensor(local_idxs)


def get_topk_predictions(
    context_pool,
    ctxt_encode_list,
    ctxt_label_uids,
    ctxt_uid_map,
    candidate_pool,
    cand_encode_list,
    cand_uid_map,
    silent,
    logger,
    top_k=10,
    is_zeshel=False,
    save_predictions=False,
):

    # handle no worlds case
    if not isinstance(cand_encode_list, dict):
        assert not isinstance(candidate_pool, dict)
        cand_encode_list = [cand_encode_list]
        candidate_pool = [candidate_pool]

    rev_cand_uid_map = {c : b for (_, b), c in cand_uid_map.items()}
    rev_ctxt_uid_map = {c : b for (_, b), c in ctxt_uid_map.items()}

    contexts = []
    context_uids = []
    pos_coref_ctxts = []
    pos_coref_ctxt_uids = []
    knn_ctxts = []
    knn_ctxt_uids = []
    pos_cands = []
    pos_cand_uids = []
    knn_cands = []
    knn_cand_uids = []

    for src in ctxt_encode_list.keys():
        contexts.append(context_pool[src])

        knn_ctxt_lids = get_knn(
            ctxt_encode_list[src],
            ctxt_encode_list[src],
            top_k+1,
            logger
        )
        knn_ctxt_lids = knn_ctxt_lids[:, 1:] # shave off self-nn for ctxt

        knn_cand_lids = get_knn(
            cand_encode_list[src],
            ctxt_encode_list[src],
            top_k,
            logger
        )

        knn_ctxts.append(context_pool[src][knn_ctxt_lids])
        knn_cands.append(candidate_pool[src][knn_cand_lids])

        ctxt_uid_mapper = np.vectorize(lambda i : ctxt_uid_map[(src, i)])
        cand_uid_mapper = np.vectorize(lambda i : cand_uid_map[(src, i)])
        knn_ctxt_uids.append(
            torch.LongTensor(ctxt_uid_mapper(knn_ctxt_lids.numpy()))
        )
        knn_cand_uids.append(
            torch.LongTensor(cand_uid_mapper(knn_cand_lids.numpy()))
        )

        label_uids = ctxt_label_uids[src].numpy().tolist()
        cand2ctxts = defaultdict(list)
        src_ctxt_uids = []
        for local_id, label_uid in enumerate(label_uids):
            _ctxt_uid = ctxt_uid_map[(src, local_id)]
            src_ctxt_uids.append(_ctxt_uid)
            cand2ctxts[label_uid].append(_ctxt_uid)
        context_uids.append(torch.LongTensor(src_ctxt_uids))

        for local_id, label_uid in enumerate(label_uids):
            coref_ctxt_uids = cand2ctxts[label_uid]
            coref_ctxt_lids = map(
                lambda i : rev_ctxt_uid_map[i],
                coref_ctxt_uids
            )
            coref_ctxt_lids = filter(lambda i: i != local_id, coref_ctxt_lids)
            coref_ctxt_lids = list(coref_ctxt_lids)
            gold_cand_lid = rev_cand_uid_map[label_uid]

            pos_coref_ctxts.append(context_pool[src][coref_ctxt_lids])
            if len(coref_ctxt_lids) > 0:
                pos_coref_ctxt_uids.append(
                    torch.LongTensor(
                        ctxt_uid_mapper(np.array(coref_ctxt_lids))
                    )
                )
            else:
                pos_coref_ctxt_uids.append(torch.LongTensor([]))
            pos_cands.append(candidate_pool[src][gold_cand_lid])
            pos_cand_uids.append(
                torch.LongTensor(cand_uid_mapper(np.asarray([gold_cand_lid])))
            )

    # structure everything as tensors except for `pos_*` variables
    contexts = torch.cat(contexts)
    context_uids = torch.cat(context_uids)
    knn_ctxts = torch.cat(knn_ctxts)
    knn_ctxt_uids = torch.cat(knn_ctxt_uids)
    knn_cands = torch.cat(knn_cands)
    knn_cand_uids = torch.cat(knn_cand_uids)

    # TODO: Compute stats
    cand_recall = defaultdict(float)
    total = len(pos_cand_uids)
    for i in range(total):
        for k in [1 << i for i in range(10)]:
            if pos_cand_uids[i].item() in knn_cand_uids[i, :k]:
                cand_recall[k] += 1.0 / total

    single_link_recall = defaultdict(float)
    avg_link_recall = defaultdict(float)
    total = 0
    for i, gold_coref_uids in enumerate(pos_coref_ctxt_uids):
        if len(gold_coref_uids) > 0:
            total += 1
            for k in [1 << i for i in range(10)]:
                _hits = [x in knn_ctxt_uids[i, :k] for x in gold_coref_uids]
                single_link_recall[k] += int(any(_hits))
                avg_link_recall[k] += np.mean(_hits)

    for k in single_link_recall.keys():
        single_link_recall[k] /= total
    for k in avg_link_recall.keys():
        avg_link_recall[k] /= total

    logger.info('entity candidate recall: {}'.format(
        json.dumps(cand_recall, sort_keys=True, indent=4)
    ))
    logger.info('single link coref recall: {}'.format(
        json.dumps(single_link_recall, sort_keys=True, indent=4)
    ))
    logger.info('avg link coref recall: {}'.format(
        json.dumps(avg_link_recall, sort_keys=True, indent=4)
    ))

    nn_data = {
        "contexts" : contexts,
        "context_uids":  context_uids,
        "pos_coref_ctxts":  pos_coref_ctxts,
        "pos_coref_ctxt_uids":  pos_coref_ctxt_uids,
        "knn_ctxts":  knn_ctxts,
        "knn_ctxt_uids":  knn_ctxt_uids,
        "pos_cands":  pos_cands,
        "pos_cand_uids":  pos_cand_uids,
        "knn_cands":  knn_cands,
        "knn_cand_uids":  knn_cand_uids,
    }
    
    return nn_data

