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


def load_entity_dict(logger, params):
    path = params.get("entity_dict_path", None)
    assert path is not None, "Error! entity_dict_path is empty."

    entity_dict = {}
    entity_json = {}
    logger.info("Loading entity description from path: " + path)
    with open(path, 'rt') as f:
        for line in f:
            sample = json.loads(line.rstrip())
            entity_id = sample['document_id']
            title = sample['title']
            text = sample.get("text", "").strip()
            entity_dict[entity_id] = (title, text)
            entity_json[entity_id] = sample
            if params["debug"] and len(entity_list) > 200:
                break

    return entity_dict, entity_json


def read_tfidf_cands(data_path, mode):
    tfidf_cand_file = os.path.join(
        data_path,
        'tfidf_candidates',
        mode + ".json"
    )
    tfidf_cand_dict = {}
    with open(tfidf_cand_file, "rt") as f:
        for line in f:
            sample = json.loads(line.rstrip())
            tfidf_cand_dict[sample["mention_id"]] = sample["tfidf_candidates"]
    return tfidf_cand_dict


def get_candidate_pool_tensor(
    entity_desc_list,
    tokenizer,
    max_seq_length,
    logger,
):
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


def main(params):
    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model 
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer

    # laod entities
    entity_dict, entity_json = load_entity_dict(logger, params)

    # load tfidf candidates
    tfidf_cand_dict = read_tfidf_cands(params["data_path"], params["mode"])

    # load mentions
    test_samples = utils.read_dataset(params["mode"], params["data_path"])
    logger.info("Read %d test samples." % len(test_samples))

    # get only the cands we need to tokenize
    cand_ids = [c for l in tfidf_cand_dict.values() for c in l]
    cand_ids.extend([x["label_umls_cuid"] for x in test_samples])
    cand_ids = list(set(cand_ids))
    num_cands = len(cand_ids)

    # tokenize the candidates
    cand_uid_map = {c : i for i, c in enumerate(cand_ids)}
    candidate_pool = get_candidate_pool_tensor(
        [entity_dict[c] for c in cand_ids],
        tokenizer,
        params["max_cand_length"],
        logger
    )

    # create mention maps
    ctxt_uid_map = {x["mm_mention_id"] : i + num_cands
                        for i, x in enumerate(test_samples)}
    ctxt_cand_map = {x["mm_mention_id"] : x["label_umls_cuid"]
                        for x in test_samples}
    ctxt_doc_map = {x["mm_mention_id"] : x["context_doc_id"]
                        for x in test_samples}
    doc_ctxt_map = defaultdict(list)
    for c, d in ctxt_doc_map.items():
        doc_ctxt_map[d].append(c)

    # create text maps for investigative evaluation
    uid_to_json = {
        uid : entity_json[cuid] for cuid, uid in cand_uid_map.items()
    }
    uid_to_json.update({i+num_cands : x for i, x in enumerate(test_samples)})

    # tokenize the contexts
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
    context_pool = test_data["context_vecs"]
    
    # create output variables
    contexts = context_pool
    context_uids = torch.LongTensor(list(ctxt_uid_map.values()))

    pos_coref_ctxts = []
    pos_coref_ctxt_uids = []
    for i, c in enumerate(ctxt_doc_map.keys()):
        assert ctxt_uid_map[c] == i + num_cands
        doc = ctxt_doc_map[c]
        coref_ctxts = [x for x in doc_ctxt_map[doc]
                          if x != c and ctxt_cand_map[x] == ctxt_cand_map[c]]
        coref_ctxt_uids = [ctxt_uid_map[x] for x in coref_ctxts]
        coref_ctxt_idxs = [x - num_cands for x in coref_ctxt_uids]
        pos_coref_ctxts.append(context_pool[coref_ctxt_idxs])
        pos_coref_ctxt_uids.append(torch.LongTensor(coref_ctxt_uids))

    knn_ctxts = []
    knn_ctxt_uids = []
    for i, c in enumerate(ctxt_doc_map.keys()):
        assert ctxt_uid_map[c] == i + num_cands
        doc = ctxt_doc_map[c]
        wdoc_ctxts = [x for x in doc_ctxt_map[doc] if x != c]
        wdoc_ctxt_uids = [ctxt_uid_map[x] for x in wdoc_ctxts]
        wdoc_ctxt_idxs = [x - num_cands for x in wdoc_ctxt_uids]
        knn_ctxts.append(context_pool[wdoc_ctxt_idxs])
        knn_ctxt_uids.append(torch.LongTensor(wdoc_ctxt_uids))
        
    pos_cands = []
    pos_cand_uids = []
    for i, c in enumerate(ctxt_cand_map.keys()):
        assert ctxt_uid_map[c] == i + num_cands
        pos_cands.append(candidate_pool[cand_uid_map[ctxt_cand_map[c]]])
        pos_cand_uids.append(torch.LongTensor([cand_uid_map[ctxt_cand_map[c]]]))

    knn_cands = []
    knn_cand_uids = []
    for i, c in enumerate(ctxt_cand_map.keys()):
        assert ctxt_uid_map[c] == i + num_cands
        tfidf_cands = tfidf_cand_dict.get(c, [])
        tfidf_cand_uids = [cand_uid_map[x] for x in tfidf_cands]
        knn_cands.append(candidate_pool[tfidf_cand_uids])
        knn_cand_uids.append(torch.LongTensor(tfidf_cand_uids))

    tfidf_data = {
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
        "uid_to_json":  uid_to_json,
    }
    
    save_data_path = os.path.join(
        params['output_path'], 
        'joint_candidates_%s_tfidf.t7' % (params['mode'])
    )
    torch.save(tfidf_data, save_data_path)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
