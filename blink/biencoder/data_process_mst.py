# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
from blink.biencoder.zeshel_utils import world_to_id
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG
from numba import jit, njit

from IPython import embed

@njit(parallel=True)
def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]

@jit(parallel=True)
def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
):
    mention_tokens = []
    if sample[mention_key] and len(sample[mention_key]) > 0:
        mention_tokens = tokenizer.tokenize(sample[mention_key])
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    context_left = sample[context_key + "_left"]
    context_right = sample[context_key + "_right"]
    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (
        context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )

    context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }

@jit(parallel=True)
def get_candidate_representation(
    candidate_desc, 
    tokenizer, 
    max_seq_length, 
    candidate_title=None,
    title_tag=ENT_TITLE_TAG,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    cand_tokens = tokenizer.tokenize(candidate_desc)
    if candidate_title is not None:
        title_tokens = tokenizer.tokenize(candidate_title)
        if len(title_tokens) <= len(cand_tokens):
            cand_tokens = title_tokens + [title_tag] + cand_tokens[(0 if title_tokens != cand_tokens[:len(title_tokens)] else len(title_tokens)):] # Filter title from description
        else:
            cand_tokens = title_tokens + [title_tag] + cand_tokens

    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }

def process_mention_data(
    samples,
    entity_dictionary,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    knn,
    dictionary_processed=False,
    mention_key="mention",
    context_key="context",
    label_key="label",
    multi_label_key=None,
    title_key='label_title',
    label_id_key='label_id',
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
):
    processed_samples = []
    dict_cui_to_idx = {}
    for idx, ent in enumerate(tqdm(entity_dictionary, desc="Tokenizing dictionary")):
        dict_cui_to_idx[ent["cui"]] = idx
        if not dictionary_processed:
            label_representation = get_candidate_representation(
                ent["description"], tokenizer, max_cand_length, ent["title"]
            )
            entity_dictionary[idx]["tokens"] = label_representation["tokens"]
            entity_dictionary[idx]["ids"] = label_representation["ids"]

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples, desc="Processing mentions")

    for idx, sample in enumerate(iter_):
        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
        )

        labels, record_labels, record_cuis = [sample], [], []
        if multi_label_key is not None:
            labels = sample[multi_label_key]
        for l in labels:
            label = l[label_key]
            label_idx = l[label_id_key]
            record_labels.append(dict_cui_to_idx[label_idx])
            record_cuis.append(label_idx)
        
        record = {
            "mention_id": sample.get("mention_id", idx),
            "mention_name": sample["mention"],
            "context": context_tokens,
            "n_labels": len(record_labels),
            "label_idxs": record_labels + [-1]*(knn - len(record_labels)), # knn-length array with the starting elements representing the ground truth, and -1 elsewhere
            "label_cuis": record_cuis,
            "type": sample["type"]
        }

        processed_samples.append(record)

    if logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )
            for l in sample["label_idxs"]:
                if l == -1:
                    break
                logger.info(f"Label {l} tokens : " + " ".join(entity_dictionary[l]["tokens"]))
                logger.info(
                    f"Label {l} ids : " + " ".join([str(v) for v in entity_dictionary[l]["ids"]])
                )

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    label_idxs = torch.tensor(
        select_field(processed_samples, "label_idxs"), dtype=torch.long,
    )
    n_labels = torch.tensor(
        select_field(processed_samples, "n_labels"), dtype=torch.int,
    )
    mention_idx = torch.arange(len(n_labels), dtype=torch.int)

    tensor_data = TensorDataset(context_vecs, label_idxs, n_labels, mention_idx)

    return processed_samples, entity_dictionary, tensor_data

@jit(parallel=True)
def compute_gold_clusters(mention_data):
    clusters = {}
    for men_idx, mention in enumerate(mention_data):
        for i in range(mention['n_labels']):
            label_idx = mention['label_idxs'][i]
            if label_idx not in clusters:
                clusters[label_idx] = []
            clusters[label_idx].append(men_idx)
    return clusters
