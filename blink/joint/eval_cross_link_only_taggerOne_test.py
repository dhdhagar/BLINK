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
    uid_to_json = eval_data['uid_to_json']
    mention_uids = [k for k, v in uid_to_json.items() if 'title' not in v.keys()]
    cuid_to_uid = {v['document_id'] : k for k, v in uid_to_json.items()
                        if 'title' in v.keys()}

    gold_linking_map = {}
    for muid in mention_uids:
        label_cuid = uid_to_json[muid]['label_umls_cuid']
        if label_cuid is not None:
            gold_linking_map[muid] = cuid_to_uid.get(label_cuid, -1)
        else:
            gold_linking_map[muid] = -1

    return gold_linking_map


def get_taggerOne_metrics(eval_data):
    uid_to_json = eval_data['uid_to_json']
    mention_uids = [k for k, v in uid_to_json.items() if 'title' not in v.keys()]
    cuid_to_type = {v['document_id'] : v['type'] for k, v in uid_to_json.items()
                        if 'title' in v.keys()}

    segment_hits, ner_hits, linking_hits = 0, 0, 0
    for muid in mention_uids:
        label_cuid = uid_to_json[muid]['label_umls_cuid'] 
        pred_cuid = uid_to_json[muid]['taggerOne_pred_umls_cuid'] 
        if label_cuid is not None:
            segment_hits += 1
            if cuid_to_type.get(label_cuid, None) == cuid_to_type.get(pred_cuid, -1):
                ner_hits += 1
                if label_cuid == pred_cuid:
                    linking_hits += 1

    num_pred_mentions = len(mention_uids)

    taggerOne_metrics = {
        'taggerOne num_pred_mentions' : num_pred_mentions,
        'taggerOne segmentation hits' : segment_hits,
        'taggerOne NER hits' : ner_hits,
        'taggerOne linking hits' : linking_hits,
        'taggerOne segmentation precision' : segment_hits / num_pred_mentions,
        'taggerOne NER precision' : ner_hits / num_pred_mentions,
        'taggerOne linking precision' : linking_hits / num_pred_mentions,
    }
    return taggerOne_metrics


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

    # get all of the edges
    cand_edges = None

    dev_cache_path = os.path.join(eval_output_path, 'taggerOne_test_cand_edges.t7')
    if not os.path.exists(dev_cache_path):
        cand_edges = score_contexts(
            cand_reranker,
            cand_dataloader,
            device=device,
            logger=logger,
            context_length=context_length,
            suffix="cand",
            silent=params["silent"],
        )
    else:
        cand_edges = torch.load(dev_cache_path)

    # construct ground truth data
    gold_linking_map = build_ground_truth(eval_data)

    # compute TaggerOne pred metrics
    taggerOne_pred_metrics = get_taggerOne_metrics(eval_data)

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
        cand_graph, gold_linking_map
    )
    logger.info('Done.')

    metrics = {
        'e2e_taggerOne_cand_gen_recall' : linking_metrics['vanilla_recall'],
        'e2e_vanilla_precision' : linking_metrics['vanilla_accuracy'],
    }
    metrics.update(taggerOne_pred_metrics)

    logger.info('metrics: {}'.format(
        json.dumps(metrics, sort_keys=True, indent=4)
    ))

    # save all of the predictions for later analysis
    save_data = {}
    save_data.update(metrics)
    save_data.update(linking_metrics)

    save_fname = os.path.join(eval_output_path, 'taggerOne_test_results.t7')
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
