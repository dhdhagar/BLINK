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


def get_seen_uids(train_data, eval_data):
    train_uid_to_json = train_data['uid_to_json']
    train_mention_uids = [
        k for k, v in train_uid_to_json.items() if 'title' not in v.keys()
    ]
    train_gold_entity_cuids = [
        train_uid_to_json[muid]['label_umls_cuid']
            for muid in train_mention_uids
    ]
    eval_uid_to_json = eval_data['uid_to_json']
    eval_mention_uids = [
        k for k, v in eval_uid_to_json.items() if 'title' not in v.keys()
    ]
    eval_cuid_to_uid = {
        v['document_id'] : k for k, v in eval_uid_to_json.items()
            if 'title' in v.keys()
    }

    seen_uids = [eval_cuid_to_uid.get(x,-1) for x in train_gold_entity_cuids]
    seen_uids = list(filter(lambda x : x != -1, seen_uids))
    return seen_uids


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

    params["pool_highlighted"] = False
    params["path_to_model"] = params["path_to_cand_model"]
    cand_reranker = CrossEncoderRanker(params)
    cand_model = cand_reranker.model

    params["path_to_model"] = None
    tokenizer = cand_reranker.tokenizer

    device = cand_reranker.device
    n_gpu = cand_reranker.n_gpu
    context_length = params["max_context_length"]

    # fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cand_reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # create eval dataloaders
    fname = os.path.join(
        params["data_path"],
        "joint_" + params["mode"] +".t7"
    )
    eval_data = torch.load(fname)
    cand_dataloader = create_eval_dataloader(
        params,
        eval_data["contexts"],
        eval_data["context_uids"],
        eval_data["knn_cands"],
        eval_data["knn_cand_uids"]
    )

    # construct ground truth data
    gold_linking_map, gold_coref_clusters = build_ground_truth(eval_data)

    # get uids we trained on
    train_data_fname = os.path.join(
        params["data_path"],
        "joint_train.t7"
    )
    train_data = torch.load(train_data_fname)
    seen_uids = get_seen_uids(train_data, eval_data)

    # get all of the edges
    cand_edges = None
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

    _cand_data = cand_edges[:, 2].cpu().numpy()
    _cand_row = cand_edges[:, 1].cpu().numpy()
    _cand_col = cand_edges[:, 0].cpu().numpy()
    cand_graph = coo_matrix(
        (_cand_data, (_cand_row, _cand_col)), shape=sparse_shape
    )

    logger.info('Computing linking metrics...')
    linking_metrics, slim_linking_graph = compute_linking_metrics(
        cand_graph, gold_linking_map, seen_uids=seen_uids
    )
    logger.info('Done.')

    uid_to_json = eval_data['uid_to_json']
    _cand_row = cand_graph.row
    _cand_col = cand_graph.col
    _cand_data = cand_graph.data
    _gt_row, _gt_col, _gt_data = [], [], []
    for r, c, d in zip(_cand_row, _cand_col, _cand_data):
        if uid_to_json[r]['type'] == uid_to_json[c]['type']:
            _gt_row.append(r)
            _gt_col.append(c)
            _gt_data.append(d)

    gold_type_cand_graph = coo_matrix(
        (_gt_data, (_gt_row, _gt_col)), shape=sparse_shape
    )

    logger.info('Computing gold-type linking metrics...')
    gold_type_linking_metrics, slim_linking_graph = compute_linking_metrics(
        gold_type_cand_graph, gold_linking_map, seen_uids=seen_uids
    )
    logger.info('Done.')

    metrics = {
        'vanilla_recall' : linking_metrics['vanilla_recall'],
        'vanilla_accuracy' : linking_metrics['vanilla_accuracy'],
        'gold_type_vanilla_accuracy' : gold_type_linking_metrics['vanilla_accuracy'],
        'seen_accuracy' : linking_metrics['seen_accuracy'],
        'unseen_accuracy' : linking_metrics['unseen_accuracy'],
        'gold_type_seen_accuracy' : gold_type_linking_metrics['seen_accuracy'],
        'gold_type_unseen_accuracy' : gold_type_linking_metrics['unseen_accuracy'],
    }

    logger.info('joint_metrics: {}'.format(
        json.dumps(metrics, sort_keys=True, indent=4)
    ))

    # save all of the predictions for later analysis
    save_data = {}
    save_data.update(linking_metrics)
    gold_type_linking_metrics = {
        'gold_type_'+k : v for k, v in gold_type_linking_metrics.items()
    }
    save_data.update(gold_type_linking_metrics)

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
