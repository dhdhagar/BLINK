# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from blink.common.params import BlinkParser
#
import os
import argparse
from datetime import datetime
import pickle
import torch
import json
import sys
import io
import random
import time
import numpy as np
import math
 
from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer

import blink.candidate_retrieval.utils
from blink.joint.crossencoder import CrossEncoderRanker
import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser
from blink.utils import copy_directory

from IPython import embed


logger = None


def modify(context_input, candidate_input, max_seq_length):
    new_input = []
    context_input = context_input.tolist()
    candidate_input = candidate_input.tolist()

    for i in range(len(context_input)):
        cur_input = context_input[i]
        cur_candidate = candidate_input[i]
        mod_input = []
        for j in range(len(cur_candidate)):
            # remove [CLS] token from candidate
            sample = cur_input + cur_candidate[j][1:]
            sample = sample[:max_seq_length]
            mod_input.append(sample)

        new_input.append(mod_input)

    return torch.LongTensor(new_input)


def evaluate(
    reranker,
    eval_dataloader,
    device,
    logger,
    context_length,
    suffix=None,
    silent=True
):

    assert suffix is not None

    reranker.model.eval()
    if silent:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    all_logits = []

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, label_input = batch
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, label_input, context_length)

        logits = logits.detach().cpu().numpy()
        label_ids = label_input.cpu().numpy()

        tmp_eval_accuracy = utils.accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        all_logits.extend(logits)

        nb_eval_examples += context_input.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("Eval accuracy (%s): %.5f" % (suffix, normalized_eval_accuracy))
    results["normalized_accuracy"] = normalized_eval_accuracy
    results["logits"] = all_logits
    return results


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def build_gold_coref_clusters(data):
    context_uids = data["context_uids"].tolist()
    uid2idx = {u : i for i, u in enumerate(context_uids)}
    # build ground truth coref clusters in terms of idxs, NOT uids
    gold_coref_clusters = [
        tuple(sorted([ctxt_uid] + coref_ctxts.tolist()))
            for ctxt_uid, coref_ctxts in zip(
                context_uids,
                data["pos_coref_ctxt_uids"]
            )
    ]
    gold_coref_clusters = [list(x) for x in set(gold_coref_clusters)]
    gold_coref_clusters = [sorted([uid2idx[x] for x in l])
                                for l in gold_coref_clusters]
    return gold_coref_clusters


def create_mst_joint_dataloader(
    params,
    gold_coref_clusters,
    contexts,
    pos_ctxts,
    pos_ctxt_uids,
    knn_ctxts,
    knn_ctxt_uids,
    pos_cands,
    pos_cand_uids,
    knn_cands,
    knn_cand_uids,
    evaluate=False
):

    max_n = None
    if evaluate:
        max_n = 2048
    if params["debug"]:
        max_n = 200
    example_bundle_size = params["example_bundle_size"]
    batch_size = params["eval_batch_size"] if evaluate \
                    else params["train_batch_size"]

    if max_n:
        gold_coref_clusters = gold_coref_clusters[:max_n]

    cluster_list_data = []

    for c in tqdm(gold_coref_clusters):
        input_examples = []
        ctxt_mask = []
        pos_mask = []
        idx_tuples = []
        for i, idx in enumerate(c):   # for all idxs in the cluster
            if len(pos_ctxts[idx]) > 0:
                input_examples.append(
                    modify(
                        contexts[idx].unsqueeze(0),
                        pos_ctxts[idx].unsqueeze(0),
                        params["max_seq_length"]
                    ).squeeze(0)
                )
                ctxt_mask.extend([True] * (len(c)-1))
                pos_mask.extend([True] * (len(c)-1))
                idx_tuples.extend([(i, x) for x in range(len(c)) if x != i])

            if knn_ctxts[idx].shape[0] > 0:
                neg_ctxts = []
                for j in range(knn_ctxts[idx].shape[0]):
                    if knn_ctxt_uids[idx][j].item() not in pos_ctxt_uids[idx]:
                        neg_ctxts.append(knn_ctxts[idx][j])
                neg_ctxts = torch.stack(neg_ctxts)
                input_examples.append(
                    modify(
                        contexts[idx].unsqueeze(0),
                        neg_ctxts.unsqueeze(0),
                        params["max_seq_length"]
                    ).squeeze(0)
                )
                ctxt_mask.extend([True] * neg_ctxts.shape[0])
                pos_mask.extend([False] * neg_ctxts.shape[0])
                idx_tuples.extend([(i, -1)] * neg_ctxts.shape[0])

            if len(pos_cands[idx]) > 0:
                _pos_cand = pos_cands[idx].unsqueeze(0)
                input_examples.append(
                    modify(
                        contexts[idx].unsqueeze(0),
                        _pos_cand.unsqueeze(0),
                        params["max_seq_length"]
                    ).squeeze(0)
                )
                ctxt_mask.append(False)
                pos_mask.append(True)
                idx_tuples.append((i, len(c)))

            if knn_cands[idx].shape[0] > 0:
                neg_cands = []
                for j in range(knn_cands[idx].shape[0]):
                    if knn_cand_uids[idx][j].item() not in pos_cand_uids[idx]:
                        neg_cands.append(knn_cands[idx][j])
                neg_cands = torch.stack(neg_cands)
                input_examples.append(
                    modify(
                        contexts[idx].unsqueeze(0),
                        neg_cands.unsqueeze(0),
                        params["max_seq_length"]
                    ).squeeze(0)
                )
                ctxt_mask.extend([False] * neg_cands.shape[0])
                pos_mask.extend([False] * neg_cands.shape[0])
                idx_tuples.extend([(i, -1)] * neg_cands.shape[0])


        input_examples = torch.cat(input_examples)
        ctxt_mask = torch.tensor(ctxt_mask, dtype=torch.bool)
        pos_mask = torch.tensor(pos_mask, dtype=torch.bool)
        idx_tuples = torch.tensor(idx_tuples, dtype=torch.long)
        
        cluster_list_data.append(
            (input_examples, ctxt_mask, pos_mask, idx_tuples)
        )

    sampler = RandomSampler(cluster_list_data)
    mst_dataloader = DataLoader(
        cluster_list_data, 
        sampler=sampler, 
        batch_size=1
    )
    return mst_dataloader


def create_dataloader(
    params,
    contexts,
    pos_cands,
    pos_cand_uids,
    knn_cands,
    knn_cand_uids,
    evaluate=False
):

    max_n = None
    if evaluate:
        max_n = 2048
    if params["debug"]:
        max_n = 200
    example_bundle_size = params["example_bundle_size"]
    batch_size = params["eval_batch_size"] if evaluate \
                    else params["train_batch_size"]

    context_input = None
    context_input_chunks = []

    for i in trange(contexts.shape[0]):
        if len(pos_cands[i]) == 0 or knn_cand_uids[i].shape[0] == 0:
            continue
        ex_pos_cands = pos_cands[i]
        if len(ex_pos_cands.shape) == 1:
            ex_pos_cands = ex_pos_cands.unsqueeze(0)
        for j in range(ex_pos_cands.shape[0]):
            candidate_bundle = ex_pos_cands[j].unsqueeze(0)
            k = 0
            while candidate_bundle.shape[0] < example_bundle_size:
                k %= knn_cand_uids[i].shape[0]
                if knn_cand_uids[i][k].item() in pos_cand_uids[i]:
                    k += 1
                    continue
                candidate_bundle = torch.cat(
                    (candidate_bundle, knn_cands[i][k].unsqueeze(0))
                )
                k += 1

            context_input_chunks.append(
                modify(
                    contexts[i].unsqueeze(0),
                    candidate_bundle.unsqueeze(0),
                    params["max_seq_length"]
                )
            )

    # concatenate all of the chunks together
    context_input = torch.cat(context_input_chunks) 
    if max_n:
        context_input = context_input[:max_n]

    # labels for each softmax bundle (positive always first)
    label_input = torch.zeros((context_input.shape[0],), dtype=torch.long)

    tensor_data = TensorDataset(context_input, label_input)
    sampler = RandomSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, 
        sampler=sampler, 
        batch_size=batch_size
    )
    return dataloader


def dual_reranker_score(
    input_ids,
    ctxt_mask,
    ctxt_reranker,
    cand_reranker,
    context_length
):
    cand_mask = ~ctxt_mask
    scores = torch.zeros_like(ctxt_mask).type(torch.float)
    if torch.sum(ctxt_mask) > 0:
        scores[ctxt_mask] = ctxt_reranker.score_candidate(
            input_ids[ctxt_mask].unsqueeze(0), context_length
        )
    if torch.sum(cand_mask) > 0:
        scores[cand_mask] = cand_reranker.score_candidate(
            input_ids[cand_mask].unsqueeze(0), context_length
        )
    return scores


def dual_reranker_forward(
    input_ids,
    label_input,
    ctxt_mask,
    ctxt_reranker,
    cand_reranker,
    context_length,
    objective
):
    batch_size, bundle_width, _ = input_ids.shape
    input_ids = input_ids.reshape(batch_size*bundle_width, -1)
    ctxt_mask = ctxt_mask.reshape(batch_size*bundle_width,)
    scores = dual_reranker_score(
        input_ids, ctxt_mask, ctxt_reranker, cand_reranker, context_length
    )
    scores = scores.reshape(batch_size, bundle_width)
    if objective == "softmax":
        loss = F.cross_entropy(scores, label_input, reduction="mean")
    else:
        assert objective == "max_margin"
        mask = torch.zeros_like(scores).type(torch.bool)
        mask[:, label_input] = True
        pos_scores = scores[mask].unsqueeze_(1)
        neg_scores = scores[~mask].reshape(scores.shape[0], -1)
        loss = torch.mean(F.relu(neg_scores - pos_scores + self.margin))
    return loss, scores


def train_one_epoch_mst_joint(
    train_dataloader,
    ctxt_reranker,
    ctxt_optimizer,
    ctxt_scheduler,
    cand_reranker,
    cand_optimizer,
    cand_scheduler,
    logger,
    params,
    epoch_idx,
    device=None,
):

    context_length = params["max_context_length"]
    grad_acc_steps = params["gradient_accumulation_steps"]
    example_bundle_size = params["example_bundle_size"]
    ctxt_model = ctxt_reranker.model
    cand_model = cand_reranker.model

    ctxt_model.train()
    cand_model.train()

    tr_loss = 0
    results = None

    if params["silent"]:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader, desc="Batch")

    for step, batch in enumerate(iter_):
        batch = tuple(t.squeeze(0) for t in batch)
        input_examples, ctxt_mask, pos_mask, idx_tuples = batch
        cluster_size = torch.max(idx_tuples)
        train_input, train_ctxt_mask = None, None
        with torch.no_grad():
            # get scores
            scores = []
            tensor_data = TensorDataset(input_examples, ctxt_mask)
            sampler = SequentialSampler(tensor_data)
            infer_dataloader = DataLoader(
                tensor_data, 
                sampler=sampler, 
                batch_size=params["eval_batch_size"]*example_bundle_size
            )
            for sub_batch in infer_dataloader:
                sub_batch = tuple(t.to(device) for t in sub_batch)
                scores.append(
                    dual_reranker_score(
                        sub_batch[0],
                        sub_batch[1],
                        ctxt_reranker,
                        cand_reranker,
                        context_length
                    ).cpu()
                )
            scores = torch.cat(scores)

            # compute mst, building necessary data structures
            pos_tuples = idx_tuples[pos_mask].t().numpy()
            inv_pos_map = {
                (a, b) : c 
                    for a, b, c in zip(
                        pos_tuples[0], pos_tuples[1], np.where(pos_mask)[0]
                    )
            }

            pos_rows, pos_cols = None, None
            if step > 5000:
                affinity_matrix = csr_matrix(
                    (-scores[pos_mask].cpu().numpy(), pos_tuples),
                    shape=tuple([cluster_size+1]*2)
                )
                mst = minimum_spanning_tree(affinity_matrix).tocoo()
                pos_rows, pos_cols = mst.row, mst.col
            else:
                pos_rows, pos_cols = pos_tuples[0], pos_tuples[1]

            # build train data
            context_bundles, context_mask_bundles = [], []
            for r, c in zip(pos_rows, pos_cols):
                pos_idx = inv_pos_map[(r, c)]
                bundle = [input_examples[pos_idx].unsqueeze(0)] 
                ctxt_mask_bundle = [ctxt_mask[pos_idx].unsqueeze(0)]
                neg_mask = (idx_tuples[:, 0] == r) & ~pos_mask

                neg_scores = scores[neg_mask]
                num_avail_negs = torch.sum(neg_mask)
                if num_avail_negs == 0:
                    continue
                neg_sample_size = min(num_avail_negs, example_bundle_size-1)
                _, hard_neg_idxs = torch.topk(
                    neg_scores, neg_sample_size
                )

                neg_input_examples = input_examples[neg_mask][hard_neg_idxs]
                neg_ctxt_mask = ctxt_mask[neg_mask][hard_neg_idxs]
                while neg_input_examples.shape[0] < example_bundle_size-1:
                    neg_input_examples = torch.cat(
                        (neg_input_examples, neg_input_examples)
                    )
                    neg_ctxt_mask = torch.cat(
                        (neg_ctxt_mask, neg_ctxt_mask)
                    )
                neg_input_examples = neg_input_examples[:example_bundle_size-1]
                neg_ctxt_mask = neg_ctxt_mask[:example_bundle_size-1]

                bundle.append(neg_input_examples)
                ctxt_mask_bundle.append(neg_ctxt_mask)
                context_bundles.append(torch.cat(bundle).unsqueeze(0))
                context_mask_bundles.append(torch.cat(ctxt_mask_bundle))
            train_input = torch.cat(context_bundles)
            train_ctxt_mask = torch.stack(context_mask_bundles)

        label_input = torch.zeros(
            (train_input.shape[0],), dtype=torch.long
        )
        train_tensor_data = TensorDataset(
            train_input, train_ctxt_mask, label_input
        )
        train_sampler = RandomSampler(train_tensor_data)
        train_dataloader = DataLoader(
            train_tensor_data, 
            sampler=train_sampler, 
            batch_size=params["train_batch_size"]
        )
        iter_ = train_dataloader
        for _, sub_batch in enumerate(iter_):
            sub_batch = tuple(t.to(device) for t in sub_batch)
            loss, _ = dual_reranker_forward(
                sub_batch[0],
                sub_batch[2],
                sub_batch[1],
                ctxt_reranker,
                cand_reranker,
                context_length,
                params["objective"]
            )
            loss.backward()
            tr_loss += loss.item() / len(iter_)

            # optimizer and scheduler for both models
            torch.nn.utils.clip_grad_norm_(
                ctxt_model.parameters(), params["max_grad_norm"]
            )
            ctxt_optimizer.step()
            ctxt_scheduler.step()
            ctxt_optimizer.zero_grad()

            torch.nn.utils.clip_grad_norm_(
                cand_model.parameters(), params["max_grad_norm"]
            )
            cand_optimizer.step()
            cand_scheduler.step()
            cand_optimizer.zero_grad()

        if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
            logger.info(
                "({}) Step {} - epoch {} average loss: {}\n".format(
                    "joint",
                    step,
                    epoch_idx,
                    tr_loss / (params["print_interval"] * grad_acc_steps),
                )
            )
            tr_loss = 0


def main(params):

    # create output dir
    now = datetime.now()
    datetime_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    program_name = os.path.splitext(os.path.basename(__file__))[0]
    if params.get("debug", False):
        model_output_path = os.path.join(
            params["output_path"], program_name, "debug"
        )
    else:
        model_output_path = os.path.join(
            params["output_path"], program_name, datetime_str
        )
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    # get logger
    logger = utils.get_logger(model_output_path)

    # copy blink source and create rerun script
    blink_copy_path = os.path.join(model_output_path, "blink")
    copy_directory("blink", blink_copy_path)
    cmd = sys.argv
    with open(os.path.join(model_output_path, "rerun.sh"), "w") as f:
        cmd.insert(0, "python")
        f.write(" ".join(cmd))

    # Init model
    ctxt_reranker = CrossEncoderRanker(params)
    ctxt_model = ctxt_reranker.model
    tokenizer = ctxt_reranker.tokenizer

    params["pool_highlighted"] = False # only `True` for ctxt
    cand_reranker = CrossEncoderRanker(params)
    cand_model = cand_reranker.model

    device = ctxt_reranker.device
    n_gpu = ctxt_reranker.n_gpu

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if ctxt_reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    context_length = params["max_context_length"]

    # create train dataloaders
    fname = os.path.join(params["data_path"], "joint_train.t7")
    train_data = torch.load(fname)
    gold_coref_clusters = build_gold_coref_clusters(train_data)
    train_dataloader = create_mst_joint_dataloader(
        params,
        gold_coref_clusters,
        train_data["contexts"],
        train_data["pos_coref_ctxts"],
        train_data["pos_coref_ctxt_uids"],
        train_data["knn_ctxts"],
        train_data["knn_ctxt_uids"],
        train_data["pos_cands"],
        train_data["pos_cand_uids"],
        train_data["knn_cands"],
        train_data["knn_cand_uids"]
    )

    fname = os.path.join(params["data_path"], "joint_valid.t7")
    valid_data = torch.load(fname)
    ctxt_valid_dataloader = create_dataloader(
        params,
        valid_data["contexts"],
        valid_data["pos_coref_ctxts"],
        valid_data["pos_coref_ctxt_uids"],
        valid_data["knn_ctxts"],
        valid_data["knn_ctxt_uids"],
        evaluate=True
    )
    cand_valid_dataloader = create_dataloader(
        params,
        valid_data["contexts"],
        valid_data["pos_cands"],
        valid_data["pos_cand_uids"],
        valid_data["knn_cands"],
        valid_data["knn_cand_uids"],
        evaluate=True
    )

    # evaluate before training
    ctxt_results = evaluate(
        ctxt_reranker,
        ctxt_valid_dataloader,
        device=device,
        logger=logger,
        context_length=context_length,
        suffix="ctxt",
        silent=params["silent"],
    )
    cand_results = evaluate(
        cand_reranker,
        cand_valid_dataloader,
        device=device,
        logger=logger,
        context_length=context_length,
        suffix="cand",
        silent=params["silent"],
    )

    number_of_samples_per_dataset = {}

    time_start = time.time()

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    ctxt_optimizer = get_optimizer(ctxt_model, params)
    ctxt_scheduler = get_scheduler(
        params,
        ctxt_optimizer,
        len(train_dataloader) * train_batch_size,
        logger
    )

    cand_optimizer = get_optimizer(cand_model, params)
    cand_scheduler = get_scheduler(
        params,
        cand_optimizer,
        len(train_dataloader) * train_batch_size,
        logger
    )

    ctxt_best_epoch_idx = -1
    ctxt_best_score = -1
    cand_best_epoch_idx = -1
    cand_best_score = -1

    num_train_epochs = params["num_train_epochs"]

    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        # train both models
        train_one_epoch_mst_joint(
            train_dataloader,
            ctxt_reranker,
            ctxt_optimizer,
            ctxt_scheduler,
            cand_reranker,
            cand_optimizer,
            cand_scheduler,
            logger,
            params,
            epoch_idx,
            device=device,
        )

        logger.info("***** Saving fine - tuned models *****")
        ctxt_epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx), "ctxt"
        )
        utils.save_model(ctxt_model, tokenizer, ctxt_epoch_output_folder_path)
        cand_epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx), "cand"
        )
        utils.save_model(cand_model, tokenizer, cand_epoch_output_folder_path)

        ctxt_results = evaluate(
            ctxt_reranker,
            ctxt_valid_dataloader,
            device=device,
            logger=logger,
            context_length=context_length,
            suffix="ctxt",
            silent=params["silent"],
        )
        cand_results = evaluate(
            cand_reranker,
            cand_valid_dataloader,
            device=device,
            logger=logger,
            context_length=context_length,
            suffix="cand",
            silent=params["silent"],
        )

        ctxt_ls = [ctxt_best_score, ctxt_results["normalized_accuracy"]]
        ctxt_li = [ctxt_best_epoch_idx, epoch_idx]
        ctxt_best_score = ctxt_ls[np.argmax(ctxt_ls)]
        ctxt_best_epoch_idx = ctxt_li[np.argmax(ctxt_ls)]

        cand_ls = [cand_best_score, cand_results["normalized_accuracy"]]
        cand_li = [cand_best_epoch_idx, epoch_idx]
        cand_best_score = cand_ls[np.argmax(cand_ls)]
        cand_best_epoch_idx = cand_li[np.argmax(cand_ls)]

        logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best models
    logger.info(
        "Best ctxt performance in epoch: {}".format(ctxt_best_epoch_idx)
    )
    best_ctxt_model_path = os.path.join(
        model_output_path, "epoch_{}".format(ctxt_best_epoch_idx), "ctxt"
    )
    logger.info(
        "Best cand performance in epoch: {}".format(cand_best_epoch_idx)
    )
    best_cand_model_path = os.path.join(
        model_output_path, "epoch_{}".format(cand_best_epoch_idx), "cand"
    )

    copy_directory(
        best_ctxt_model_path,
        os.path.join(model_output_path, "best_epoch", "ctxt")
    )
    copy_directory(
        best_cand_model_path,
        os.path.join(model_output_path, "best_epoch", "cand")
    )


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_joint_train_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
