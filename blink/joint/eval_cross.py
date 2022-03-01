import os
import sys
import json
import random
import numpy as np
import torch
from tqdm import tqdm, trange

from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

import blink.candidate_ranking.utils as utils
from blink.common.params import BlinkParser
from blink.joint.crossencoder import CrossEncoderRanker
from blink.joint.joint_eval.evaluation import (
        compute_coref_metrics,
        compute_linking_metrics,
        compute_joint_metrics,
        _get_global_maximum_spanning_tree
)

from IPython import embed


logger = None


def eval_modify(context_input, candidate_input, max_seq_length):
    cur_input = context_input.tolist()
    cur_candidate = candidate_input.tolist()
    mod_input = []
    for i in range(len(cur_candidate)):
        # remove [CLS] token from candidate
        sample = cur_input + cur_candidate[i][1:]
        sample = sample[:max_seq_length]
        mod_input.append(sample)
    return torch.LongTensor(mod_input)


def create_eval_dataloader(
    params,
    contexts,
    context_uids,
    knn_cands,
    knn_cand_uids,
):
    context_input_examples = []
    example_uid_pairs = []
    for i in trange(contexts.shape[0]):
        if knn_cand_uids[i].shape[0] == 0:
            continue
        context_input_examples.append(
            eval_modify(
                contexts[i],
                knn_cands[i],
                params["max_seq_length"]
            )
        )
        example_uid_pairs.extend(
            [(context_uids[i], cand_uid) for cand_uid in knn_cand_uids[i]]
        )

    # concatenate all of the examples together
    context_input = torch.cat(context_input_examples)
    uid_pairs = torch.LongTensor(example_uid_pairs)
    assert context_input.shape[0] == uid_pairs.shape[0]

    if params["debug"]:
        max_n = 6400
        context_input = context_input[:max_n]
        uid_pairs = context_input[:max_n]

    tensor_data = TensorDataset(context_input, uid_pairs)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, 
        sampler=sampler, 
        batch_size=params["encode_batch_size"]
    )
    return dataloader


def build_ground_truth(eval_data):
    # build gold linking map
    zipped_gold_map = zip(
        eval_data["context_uids"].tolist(),
        map(lambda x : x.item(), eval_data["pos_cand_uids"])
    )
    gold_linking_map = {ctxt_uid : cand_uid
                            for ctxt_uid, cand_uid in zipped_gold_map}

    # build ground truth coref clusters
    gold_coref_clusters = [
        tuple(sorted([ctxt_uid] + coref_ctxts.tolist()))
            for ctxt_uid, coref_ctxts in zip(
                eval_data["context_uids"].tolist(),
                eval_data["pos_coref_ctxt_uids"]
            )
    ]
    gold_coref_clusters = [list(x) for x in set(gold_coref_clusters)]

    return gold_linking_map, gold_coref_clusters


def score_contexts(
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
        iter_ = tqdm(eval_dataloader, desc="scoring {} contexts".format(suffix))

    with torch.no_grad():
        edge_vertices, edge_scores = [], []
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context_input, uid_pair = batch
            edge_vertices.append(uid_pair)
            scores = reranker.score_candidate(
                context_input.unsqueeze(0),
                context_length
            )
            scores = scores.squeeze(0)
            edge_scores.append(scores)

    edge_vertices = torch.cat(edge_vertices).type(torch.float)
    edge_scores = torch.cat(edge_scores).unsqueeze(1)
    edges = torch.cat((edge_vertices, edge_scores), 1) 
    return edges

def main(params):

    # create output dir
    eval_output_path = os.path.join(params["output_path"], "eval_output")
    if not os.path.exists(eval_output_path):
        os.makedirs(eval_output_path)
    # get logger
    logger = utils.get_logger(eval_output_path)

    # output command ran
    cmd = sys.argv
    cmd.insert(0, "python")
    logger.info(" ".join(cmd))

    # load the models
    assert params["path_to_model"] is None
    params["path_to_model"] = params["path_to_ctxt_model"]
    ctxt_reranker = CrossEncoderRanker(params)
    ctxt_model = ctxt_reranker.model

    params["pool_highlighted"] = False
    params["path_to_model"] = params["path_to_cand_model"]
    cand_reranker = CrossEncoderRanker(params)
    cand_model = cand_reranker.model

    params["path_to_model"] = None
    tokenizer = ctxt_reranker.tokenizer

    device = ctxt_reranker.device
    n_gpu = ctxt_reranker.n_gpu
    context_length = params["max_context_length"]

    # fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if ctxt_reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # create eval dataloaders
    fname = os.path.join(
        params["data_path"],
        "joint_" + params["mode"] +".t7"
    )
    eval_data = torch.load(fname)
    ctxt_dataloader = create_eval_dataloader(
        params,
        eval_data["contexts"],
        eval_data["context_uids"],
        eval_data["knn_ctxts"],
        eval_data["knn_ctxt_uids"]
    )
    cand_dataloader = create_eval_dataloader(
        params,
        eval_data["contexts"],
        eval_data["context_uids"],
        eval_data["knn_cands"],
        eval_data["knn_cand_uids"]
    )

    # construct ground truth data
    gold_linking_map, gold_coref_clusters = build_ground_truth(eval_data)

    # get all of the edges
    ctxt_edges, cand_edges = None, None
    ctxt_edges = score_contexts(
        ctxt_reranker,
        ctxt_dataloader,
        device=device,
        logger=logger,
        context_length=context_length,
        suffix="ctxt",
        silent=params["silent"],
    )
    cand_edges = score_contexts(
        cand_reranker,
        cand_dataloader,
        device=device,
        logger=logger,
        context_length=context_length,
        suffix="cand",
        silent=params["silent"],
    )

    # construct the sparse graphs
    sparse_shape = tuple(2*[max(gold_linking_map.keys())+1])

    _ctxt_data = ctxt_edges[:, 2].cpu().numpy()
    _ctxt_row = ctxt_edges[:, 0].cpu().numpy()
    _ctxt_col = ctxt_edges[:, 1].cpu().numpy()
    ctxt_graph = coo_matrix(
        (_ctxt_data, (_ctxt_row, _ctxt_col)), shape=sparse_shape
    )

    _cand_data = cand_edges[:, 2].cpu().numpy()
    _cand_row = cand_edges[:, 1].cpu().numpy()
    _cand_col = cand_edges[:, 0].cpu().numpy()
    cand_graph = coo_matrix(
        (_cand_data, (_cand_row, _cand_col)), shape=sparse_shape
    )

    logger.info('Computing coref metrics...')
    coref_metrics = compute_coref_metrics(
        gold_coref_clusters, ctxt_graph
    )
    logger.info('Done.')

    logger.info('Computing linking metrics...')
    linking_metrics, slim_linking_graph = compute_linking_metrics(
        cand_graph, gold_linking_map
    )
    logger.info('Done.')

    logger.info('Computing joint metrics...')
    slim_coref_graph = _get_global_maximum_spanning_tree([ctxt_graph])
    joint_metrics = compute_joint_metrics(
        [slim_coref_graph, slim_linking_graph],
        gold_linking_map,
        min(gold_linking_map.keys())
    )
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

    logger.info('joint_metrics: {}'.format(
        json.dumps(metrics, sort_keys=True, indent=4)
    ))

    # save all of the predictions for later analysis
    save_data = {}
    save_data.update(coref_metrics)
    save_data.update(linking_metrics)
    save_data.update(joint_metrics)

    save_fname = os.path.join(eval_output_path, 'results.t7')
    torch.save(save_data, save_fname)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()
    parser.add_joint_train_args()
    parser.add_joint_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
