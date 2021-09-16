# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from collections import defaultdict
import json
import logging
import os
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.biencoder.biencoder import BiEncoderRanker
import blink.biencoder.data_process as data
import blink.joint.nn_prediction as nnquery
import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS, load_entity_dict_zeshel, Stats
from blink.common.params import BlinkParser

from IPython import embed


def load_entity_dict(logger, params, is_zeshel):
    if is_zeshel:
        return load_entity_dict_zeshel(logger, params)

    path = params.get("entity_dict_path", None)
    assert path is not None, "Error! entity_dict_path is empty."

    entity_list = []
    logger.info("Loading entity description from path: " + path)
    with open(path, 'rt') as f:
        for line in f:
            sample = json.loads(line.rstrip())
            title = sample['title']
            text = sample.get("text", "").strip()
            entity_list.append((title, text))
            if params["debug"] and len(entity_list) > 200:
                break

    return entity_list


# zeshel version of get candidate_pool_tensor
def get_candidate_pool_tensor_zeshel(
    entity_dict,
    tokenizer,
    max_seq_length,
    logger,
):
    candidate_pool = {}
    for src in range(len(WORLDS)):
        if entity_dict.get(src, None) is None:
            continue
        logger.info("Get candidate desc to id for pool %s" % WORLDS[src])
        candidate_pool[src] = get_candidate_pool_tensor(
            entity_dict[src],
            tokenizer,
            max_seq_length,
            logger,
        )

    return candidate_pool


def get_candidate_pool_tensor_helper(
    entity_desc_list,
    tokenizer,
    max_seq_length,
    logger,
    is_zeshel,
):
    if is_zeshel:
        return get_candidate_pool_tensor_zeshel(
            entity_desc_list,
            tokenizer,
            max_seq_length,
            logger,
        )
    else:
        return get_candidate_pool_tensor(
            entity_desc_list,
            tokenizer,
            max_seq_length,
            logger,
        )


def get_candidate_pool_tensor(
    entity_desc_list,
    tokenizer,
    max_seq_length,
    logger,
):
    # TODO: add multiple thread process
    logger.info("Convert candidate text to id")
    cand_pool = [] 
    for entity_desc in tqdm(entity_desc_list):
        if type(entity_desc) is tuple:
            title, entity_text = entity_desc
        else:
            title = None
            entity_text = entity_desc

        rep = data.get_candidate_representation(
                entity_text, 
                tokenizer, 
                max_seq_length,
                title,
        )
        cand_pool.append(rep["ids"])

    cand_pool = torch.LongTensor(cand_pool) 
    return cand_pool


def encode_candidate(
    reranker,
    candidate_pool,
    encode_batch_size,
    silent,
    logger,
    is_zeshel,
):
    if is_zeshel:
        src = 0
        cand_encode_dict = {}
        for src, cand_pool in candidate_pool.items():
            logger.info("Encoding candidate pool %s" % WORLDS[src])
            cand_pool_encode = encode_candidate(
                reranker,
                cand_pool,
                encode_batch_size,
                silent,
                logger,
                False,
            )
            cand_encode_dict[src] = cand_pool_encode
        return cand_encode_dict
        
    reranker.model.eval()
    device = reranker.device
    sampler = SequentialSampler(candidate_pool)
    data_loader = DataLoader(
        candidate_pool, sampler=sampler, batch_size=encode_batch_size
    )
    if silent:
        iter_ = data_loader
    else:
        iter_ = tqdm(data_loader)

    cand_encode_list = None
    for step, batch in enumerate(iter_):
        cands = batch
        cands = cands.to(device)
        cand_encode = reranker.encode_candidate(cands)
        if cand_encode_list is None:
            cand_encode_list = cand_encode.cpu()
        else:
            cand_encode_list = torch.cat((cand_encode_list, cand_encode.cpu()))

    return cand_encode_list


def load_or_generate_candidate_pool(
    tokenizer,
    params,
    logger,
    cand_pool_path,
):
    candidate_pool = None
    is_zeshel = params.get("zeshel", None)
    if cand_pool_path is not None:
        # try to load candidate pool from file
        try:
            logger.info("Loading pre-generated candidate pool from: ")
            logger.info(cand_pool_path)
            candidate_pool = torch.load(cand_pool_path)
        except:
            logger.info("Loading failed. Generating candidate pool")

    if candidate_pool is None:
        # compute candidate pool from entity list
        entity_desc_list = load_entity_dict(logger, params, is_zeshel)
        candidate_pool = get_candidate_pool_tensor_helper(
            entity_desc_list,
            tokenizer,
            params["max_cand_length"],
            logger,
            is_zeshel,
        )

        if cand_pool_path is not None:
            logger.info("Saving candidate pool.")
            torch.save(candidate_pool, cand_pool_path)

    return candidate_pool


def encode_context(
    reranker,
    context_dataloader,
    cand_uid_map,
    encode_batch_size,
    silent,
    logger,
    is_zeshel,
):
    reranker.model.eval()
    device = reranker.device
    if silent:
        iter_ = context_dataloader
    else:
        iter_ = tqdm(context_dataloader)

    context_input_list = None
    context_encode_list = None
    src_list = None
    label_id_list = None
    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, _, srcs, label_ids = batch
        context_encode = reranker.encode_context(context_input)
        if context_encode_list is None:
            context_input_list = context_input.cpu()
            context_encode_list = context_encode.cpu()
            src_list = srcs.cpu()
            label_id_list = label_ids.cpu()
        else:
            context_input_list = torch.cat(
                    (context_input_list, context_input.cpu())
            )
            context_encode_list = torch.cat(
                    (context_encode_list, context_encode.cpu())
            )
            src_list = torch.cat((src_list, srcs.cpu()))
            label_id_list = torch.cat((label_id_list, label_ids.cpu()))

    # label contexts with cand uids
    srcs = src_list.squeeze(1).numpy().tolist()
    label_ids = label_id_list.squeeze(1).numpy().tolist()
    label_uids = [cand_uid_map[k] for k in zip(srcs, label_ids)]

    # structure dictionaries
    context_input_dict = defaultdict(list)
    context_encode_dict = defaultdict(list)
    context_label_uids = defaultdict(list)
    _iter = zip(srcs, context_input_list, context_encode_list, label_uids)
    for src, ctxt_ids, ctxt_vec, lbl_uid in _iter:
        context_input_dict[src].append(ctxt_ids)
        context_encode_dict[src].append(ctxt_vec)
        context_label_uids[src].append(lbl_uid)

    for src in list(context_encode_dict.keys()):
        context_input_dict[src] = torch.stack(tuple(context_input_dict[src]))
        context_encode_dict[src] = torch.stack(tuple(context_encode_dict[src]))
        context_label_uids[src] = torch.tensor(context_label_uids[src])

    context_input_dict = dict(context_input_dict)
    context_encode_dict = dict(context_encode_dict)
    context_label_uids = dict(context_label_uids)
    return context_input_dict, context_encode_dict, context_label_uids


def get_uid_map(
    encoding,
    start_offset=0,
    is_zeshel=False
):
    if isinstance(encoding, dict):
        uid = start_offset
        uid_map = {}
        for src, vecs in encoding.items():
            uid_map.update({(src, i) : uid+i for i in range(vecs.shape[0])})
            uid = len(uid_map.keys()) + start_offset
    else:
        assert not is_zeshel
        uid_map = {(0, i) : i+start_offset for i in range(len(encoding))}

    return uid_map


def main(params):
    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model 
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model
    
    device = reranker.device
    
    # candidate encoding is not pre-computed. 
    # load/generate candidate pool to compute candidate encoding.
    cand_pool_path = params.get("cand_pool_path", None)
    candidate_pool = load_or_generate_candidate_pool(
        tokenizer,
        params,
        logger,
        cand_pool_path,
    )       

    cand_encode_path = params.get("cand_encode_path", None)
    
    candidate_encoding = None
    if cand_encode_path is not None:
        # try to load candidate encoding from path
        # if success, avoid computing candidate encoding
        try:
            logger.info("Loading pre-generated candidate encode path.")
            candidate_encoding = torch.load(cand_encode_path)
        except:
            logger.info("Loading failed. Generating candidate encoding.")

    if candidate_encoding is None:
        candidate_encoding = encode_candidate(
            reranker,
            candidate_pool,
            params["encode_batch_size"],
            silent=params["silent"],
            logger=logger,
            is_zeshel = params.get("zeshel", None)
        )

        if cand_encode_path is not None:
            # Save candidate encoding to avoid re-compute
            logger.info("Saving candidate encoding to file " + cand_encode_path)
            torch.save(candidate_encoding, cand_encode_path)

    test_samples = utils.read_dataset(params["mode"], params["data_path"])
    logger.info("Read %d test samples." % len(test_samples))

    # get candidate uid map
    cand_uid_map = get_uid_map(
            candidate_encoding,
            start_offset=0,
            is_zeshel=params.get("zeshel", None)
    )
   
    # generate context embeddings
    test_data, test_tensor_data = data.process_mention_data(
        test_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params['context_key'],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )
    test_sampler = SequentialSampler(test_tensor_data)
    test_dataloader = DataLoader(
        test_tensor_data, 
        sampler=test_sampler, 
        batch_size=params["encode_batch_size"]
    )
    context_pool, context_encoding, context_label_uids = encode_context(
        reranker,
        test_dataloader,
        cand_uid_map,
        params["encode_batch_size"],
        silent=params["silent"],
        logger=logger,
        is_zeshel = params.get("zeshel", None)
    )

    ctxt_uid_map = get_uid_map(
            context_encoding,
            start_offset=len(cand_uid_map),
            is_zeshel=params.get("zeshel", None)
    )

    save_results = params.get("save_topk_result")
    new_data = nnquery.get_topk_predictions(
        context_pool,
        context_encoding,
        context_label_uids,
        ctxt_uid_map,
        candidate_pool,
        candidate_encoding,
        cand_uid_map,
        params["silent"],
        logger,
        params["top_k"],
        params.get("zeshel", None),
        save_results,
    )

    if save_results: 
        save_data_path = os.path.join(
            params['output_path'], 
            'joint_candidates_%s_top%d.t7' % (params['mode'], params['top_k'])
        )
        torch.save(new_data, save_data_path)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
