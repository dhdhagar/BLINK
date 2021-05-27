# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import random
import time
import pickle
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from pytorch_transformers.optimization import WarmupLinearSchedule
from tqdm import tqdm, trange

import blink.biencoder.data_process_mult as data_process
import blink.biencoder.eval_cluster_linking as eval_cluster_linking
import blink.candidate_ranking.utils as utils
from blink.biencoder.biencoder import BiEncoderRanker
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser

from IPython import embed


logger = None

def evaluate(reranker, valid_dict_vecs, valid_men_vecs, device, logger, knn, n_gpu, entity_data, query_data, silent=False, use_types=False, embed_batch_size=768, force_exact_search=False, probe_mult_factor=1):
    torch.cuda.empty_cache()

    reranker.model.eval()
    n_entities = len(valid_dict_vecs)
    n_mentions = len(valid_men_vecs)
    joint_graphs = {}
    max_knn = 4
    for k in [0, 1, 2, 4]:
        joint_graphs[k] = {
            'rows': np.array([]),
            'cols': np.array([]),
            'data': np.array([]),
            'shape': (n_entities+n_mentions, n_entities+n_mentions)
        }

    if use_types:
        logger.info("Eval: Dictionary: Embedding and building index")
        dict_embeds, dict_indexes, dict_idxs_by_type = data_process.embed_and_index(reranker, valid_dict_vecs, encoder_type="candidate", n_gpu=n_gpu, corpus=entity_data, force_exact_search=force_exact_search, batch_size=embed_batch_size, probe_mult_factor=probe_mult_factor)
        logger.info("Eval: Queries: Embedding and building index")
        men_embeds, men_indexes, men_idxs_by_type = data_process.embed_and_index(reranker, valid_men_vecs, encoder_type="context", n_gpu=n_gpu, corpus=query_data, force_exact_search=force_exact_search, batch_size=embed_batch_size, probe_mult_factor=probe_mult_factor)
    else:
        logger.info("Eval: Dictionary: Embedding and building index")
        dict_embeds, dict_index = data_process.embed_and_index(
            reranker, valid_dict_vecs, 'candidate', n_gpu=n_gpu, force_exact_search=force_exact_search, batch_size=embed_batch_size, probe_mult_factor=probe_mult_factor)
        logger.info("Eval: Queries: Embedding and building index")
        men_embeds, men_index = data_process.embed_and_index(
            reranker, valid_men_vecs, 'context', n_gpu=n_gpu, force_exact_search=force_exact_search, batch_size=embed_batch_size, probe_mult_factor=probe_mult_factor)
    
    logger.info("Eval: Starting KNN search...")
    # Fetch recall_k (default 16) knn entities for all mentions
    # Fetch (k+1) NN mention candidates
    if not use_types:
        nn_ent_dists, nn_ent_idxs = dict_index.search(men_embeds, 1)
        nn_men_dists, nn_men_idxs = men_index.search(men_embeds, max_knn + 1)
    else:
        nn_ent_idxs = np.zeros((len(men_embeds), 1))
        nn_ent_dists = np.zeros((len(men_embeds), 1), dtype='float64')
        nn_men_idxs = np.zeros((len(men_embeds), max_knn + 1))
        nn_men_dists = np.zeros((len(men_embeds), max_knn + 1), dtype='float64')
        for entity_type in men_indexes:
            men_embeds_by_type = men_embeds[men_idxs_by_type[entity_type]]
            nn_ent_dists_by_type, nn_ent_idxs_by_type = dict_indexes[entity_type].search(men_embeds_by_type, 1)
            nn_men_dists_by_type, nn_men_idxs_by_type = men_indexes[entity_type].search(men_embeds_by_type, max_knn + 1)
            nn_ent_idxs_by_type = np.array(list(map(lambda x: dict_idxs_by_type[entity_type][x], nn_ent_idxs_by_type)))
            nn_men_idxs_by_type = np.array(list(map(lambda x: men_idxs_by_type[entity_type][x], nn_men_idxs_by_type)))
            for i,idx in enumerate(men_idxs_by_type[entity_type]):
                nn_ent_idxs[idx] = nn_ent_idxs_by_type[i]
                nn_ent_dists[idx] = nn_ent_dists_by_type[i]
                nn_men_idxs[idx] = nn_men_idxs_by_type[i]
                nn_men_dists[idx] = nn_men_dists_by_type[i]
    logger.info("Eval: Search finished")
    
    logger.info('Eval: Building graphs')
    for men_query_idx, men_embed in enumerate(tqdm(men_embeds, total=len(men_embeds), desc="Eval: Building graphs")):
        # Get nearest entity candidate
        dict_cand_idx = nn_ent_idxs[men_query_idx][0]
        dict_cand_score = nn_ent_dists[men_query_idx][0]
        
        # Filter candidates to remove mention query and keep only the top k candidates
        men_cand_idxs = nn_men_idxs[men_query_idx]
        men_cand_scores = nn_men_dists[men_query_idx]
        
        filter_mask = men_cand_idxs != men_query_idx
        men_cand_idxs, men_cand_scores = men_cand_idxs[filter_mask][:max_knn], men_cand_scores[filter_mask][:max_knn]

        # Add edges to the graphs
        for k in joint_graphs:
            joint_graph = joint_graphs[k]
            # Add mention-entity edge
            joint_graph['rows'] = np.append(
                joint_graph['rows'], [n_entities+men_query_idx])  # Mentions added at an offset of maximum entities
            joint_graph['cols'] = np.append(
                joint_graph['cols'], dict_cand_idx)
            joint_graph['data'] = np.append(
                joint_graph['data'], dict_cand_score)
            if k > 0:
                # Add mention-mention edges
                joint_graph['rows'] = np.append(
                    joint_graph['rows'], [n_entities+men_query_idx]*len(men_cand_idxs[:k]))
                joint_graph['cols'] = np.append(
                    joint_graph['cols'], n_entities+men_cand_idxs[:k])
                joint_graph['data'] = np.append(
                    joint_graph['data'], men_cand_scores[:k])
    
    max_eval_acc = -1.
    for k in joint_graphs:
        logger.info(f"\nEval: Graph (k={k}):")
        # Partition graph based on cluster-linking constraints
        partitioned_graph, clusters = eval_cluster_linking.partition_graph(
            joint_graphs[k], n_entities, directed=True, return_clusters=True)
        # Infer predictions from clusters
        result = eval_cluster_linking.analyzeClusters(clusters, entity_data, query_data, k)
        acc = float(result['accuracy'].split(' ')[0])
        max_eval_acc = max(acc, max_eval_acc)
        logger.info(f"Eval: accuracy for graph@k={k}: {acc}%")
    logger.info(f"Eval: Best accuracy: {max_eval_acc}%")
    return max_eval_acc, {'dict_embeds': dict_embeds, 'dict_indexes': dict_indexes, 'dict_idxs_by_type': dict_idxs_by_type} if use_types else {'dict_embeds': dict_embeds, 'dict_index': dict_index}

# ARCHIVED: The evaluate function makes a prediction on a set of knn candidates for every mention
def evaluate_ind_pred(
    reranker, valid_dataloader, valid_dict_vecs, params, device, logger, knn, n_gpu, entity_data, query_data, use_types=False, embed_batch_size=768
):
    reranker.model.eval()
    knn = max(16, 2*knn) # Accomodate the approximate-nature of the knn procedure by retrieving more samples and then filtering
    iter_ = valid_dataloader if params["silent"] else tqdm(valid_dataloader, desc="Evaluation")
    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    if not use_types:
        valid_dict_embeddings, valid_dict_index = data_process.embed_and_index(reranker, valid_dict_vecs, encoder_type="candidate", n_gpu=n_gpu, batch_size=embed_batch_size)
    else:
        valid_dict_embeddings, valid_dict_indexes, dict_idxs_by_type = data_process.embed_and_index(reranker, valid_dict_vecs, encoder_type="candidate", n_gpu=n_gpu, corpus=entity_data, batch_size=embed_batch_size)

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_inputs, candidate_idxs, n_gold, mention_idxs = batch
        
        with torch.no_grad():
            mention_embeddings = reranker.encode_context(context_inputs)
            # context_inputs: Shape: batch x token_len
            candidate_inputs = np.array([], dtype=np.int) # Shape: (batch*knn) x token_len
            label_inputs = torch.zeros((context_inputs.shape[0], knn), dtype=torch.float32) # Shape: batch x knn

            for i, m_embed in enumerate(mention_embeddings):
                if use_types:
                    entity_type = query_data[mention_idxs[i]]['type']
                    valid_dict_index = valid_dict_indexes[entity_type]
                _, knn_dict_idxs = valid_dict_index.search(np.expand_dims(m_embed, axis=0), knn)
                knn_dict_idxs = knn_dict_idxs.astype(np.int64).flatten()
                if use_types:
                    # Map type-specific indices to the entire dictionary
                    knn_dict_idxs = list(map(lambda x: dict_idxs_by_type[entity_type][x], knn_dict_idxs))
                gold_idxs = candidate_idxs[i][:n_gold[i]].cpu()
                candidate_inputs = np.concatenate((candidate_inputs, knn_dict_idxs))
                label_inputs[i] = torch.tensor([1 if nn in gold_idxs else 0 for nn in knn_dict_idxs])
            candidate_inputs = torch.tensor(list(map(lambda x: valid_dict_vecs[x].numpy(), candidate_inputs))).cuda()
            context_inputs = context_inputs.cuda()
            label_inputs = label_inputs.cuda()
            
            logits = reranker(context_inputs, candidate_inputs, label_inputs, only_logits=True)

        logits = logits.detach().cpu().numpy()
        tmp_eval_accuracy = int(torch.sum(label_inputs[np.arange(label_inputs.shape[0]), np.argmax(logits, axis=1)] == 1))
        eval_accuracy += tmp_eval_accuracy
        nb_eval_examples += context_inputs.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    return normalized_eval_accuracy

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


def main(params):
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    pickle_src_path = params["pickle_src_path"]
    if pickle_src_path is None or not os.path.exists(pickle_src_path):
        pickle_src_path = model_output_path

    knn = params["knn"]
    use_types = params["use_types"]

    # Init model
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    device = reranker.device
    n_gpu = reranker.n_gpu

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    entity_dictionary_loaded = False
    entity_dictionary_pkl_path = os.path.join(pickle_src_path, 'entity_dictionary.pickle')
    if os.path.isfile(entity_dictionary_pkl_path):
        print("Loading stored processed entity dictionary...")
        with open(entity_dictionary_pkl_path, 'rb') as read_handle:
            entity_dictionary = pickle.load(read_handle)
        entity_dictionary_loaded = True
    if not params["only_evaluate"]:
        # Load train data
        train_tensor_data_pkl_path = os.path.join(pickle_src_path, 'train_tensor_data.pickle')
        train_processed_data_pkl_path = os.path.join(pickle_src_path, 'train_processed_data.pickle')
        if os.path.isfile(train_tensor_data_pkl_path) and os.path.isfile(train_processed_data_pkl_path):
            print("Loading stored processed train data...")
            with open(train_tensor_data_pkl_path, 'rb') as read_handle:
                train_tensor_data = pickle.load(read_handle)
            with open(train_processed_data_pkl_path, 'rb') as read_handle:
                train_processed_data = pickle.load(read_handle)
        else:
            train_samples = utils.read_dataset("train", params["data_path"])
            if not entity_dictionary_loaded:
                with open(os.path.join(params["data_path"], 'dictionary.pickle'), 'rb') as read_handle:
                    entity_dictionary = pickle.load(read_handle)

            # Check if dataset has multiple ground-truth labels
            mult_labels = "labels" in train_samples[0].keys()
            if params["filter_unlabeled"]:
                # Filter samples without gold entities
                train_samples = list(filter(lambda sample: (len(sample["labels"]) > 0) if mult_labels else (sample["label"] is not None), train_samples))
            logger.info("Read %d train samples." % len(train_samples))

            train_processed_data, entity_dictionary, train_tensor_data = data_process.process_mention_data(
                train_samples,
                entity_dictionary,
                tokenizer,
                params["max_context_length"],
                params["max_cand_length"],
                context_key=params["context_key"],
                multi_label_key="labels" if mult_labels else None,
                silent=params["silent"],
                logger=logger,
                debug=params["debug"],
                knn=knn,
                dictionary_processed=entity_dictionary_loaded
            )
            print("Saving processed train data...")
            if not entity_dictionary_loaded:
                with open(entity_dictionary_pkl_path, 'wb') as write_handle:
                    pickle.dump(entity_dictionary, write_handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
            with open(train_tensor_data_pkl_path, 'wb') as write_handle:
                pickle.dump(train_tensor_data, write_handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            with open(train_processed_data_pkl_path, 'wb') as write_handle:
                pickle.dump(train_processed_data, write_handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        # Store the query mention vectors
        train_men_vecs = train_tensor_data[:][0]

        if params["shuffle"]:
            train_sampler = RandomSampler(train_tensor_data)
        else:
            train_sampler = SequentialSampler(train_tensor_data)

        train_dataloader = DataLoader(
            train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
        )
    
    # Store the entity dictionary vectors
    entity_dict_vecs = torch.tensor(list(map(lambda x: x['ids'], entity_dictionary)), dtype=torch.long)

    # Load eval data
    valid_tensor_data_pkl_path = os.path.join(pickle_src_path, 'valid_tensor_data.pickle')
    valid_processed_data_pkl_path = os.path.join(pickle_src_path, 'valid_processed_data.pickle')
    if os.path.isfile(valid_tensor_data_pkl_path) and os.path.isfile(valid_processed_data_pkl_path):
        print("Loading stored processed valid data...")
        with open(valid_tensor_data_pkl_path, 'rb') as read_handle:
            valid_tensor_data = pickle.load(read_handle)
        with open(valid_processed_data_pkl_path, 'rb') as read_handle:
            valid_processed_data = pickle.load(read_handle)
    else:
        valid_samples = utils.read_dataset("valid", params["data_path"])
        # Check if dataset has multiple ground-truth labels
        mult_labels = "labels" in valid_samples[0].keys()
        # Filter samples without gold entities
        valid_samples = list(filter(lambda sample: (len(sample["labels"]) > 0) if mult_labels else (sample["label"] is not None), valid_samples))
        logger.info("Read %d valid samples." % len(valid_samples))

        valid_processed_data, _, valid_tensor_data = data_process.process_mention_data(
            valid_samples,
            entity_dictionary,
            tokenizer,
            params["max_context_length"],
            params["max_cand_length"],
            context_key=params["context_key"],
            multi_label_key="labels" if mult_labels else None,
            silent=params["silent"],
            logger=logger,
            debug=params["debug"],
            knn=knn,
            dictionary_processed=True
        )
        print("Saving processed valid data...")
        with open(valid_tensor_data_pkl_path, 'wb') as write_handle:
            pickle.dump(valid_tensor_data, write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(valid_processed_data_pkl_path, 'wb') as write_handle:
                pickle.dump(valid_processed_data, write_handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    # Store the query mention vectors
    valid_men_vecs = valid_tensor_data[:][0]
    
    # valid_sampler = SequentialSampler(valid_tensor_data)
    # valid_dataloader = DataLoader(
    #     valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    # )

    if params["only_evaluate"]:
        evaluate(
            reranker, entity_dict_vecs, valid_men_vecs, device=device, logger=logger, knn=knn, n_gpu=n_gpu, entity_data=entity_dictionary, query_data=valid_processed_data, silent=params["silent"], use_types=use_types or params["use_types_for_eval"], embed_batch_size=params['embed_batch_size'], force_exact_search=use_types or params["use_types_for_eval"] or params["force_exact_search"], probe_mult_factor=params['probe_mult_factor']
        )
        exit()

    time_start = time.time()
    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )
    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, data_parallel: {}".format(device, n_gpu, params["data_parallel"])
    )

    # Set model to training mode
    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)
    best_epoch_idx = -1
    best_score = -1
    num_train_epochs = params["num_train_epochs"]
    
    init_base_model_run = True if params.get("path_to_model", None) is None else False
    init_run_pkl_path = os.path.join(pickle_src_path, f'init_run_{"type" if use_types else "notype"}.t7')

    dict_embed_data = None

    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        model.train()
        torch.cuda.empty_cache()
        tr_loss = 0
        results = None

        # Check if embeddings and index can be loaded
        init_run_data_loaded = False
        if init_base_model_run:
            if os.path.isfile(init_run_pkl_path):
                logger.info('Loading init run data')
                init_run_data = torch.load(init_run_pkl_path)
                init_run_data_loaded = True
        load_stored_data = init_base_model_run and init_run_data_loaded

        # Compute mention and entity embeddings at the start of each epoch
        if use_types:
            if load_stored_data:
                train_dict_embeddings, dict_idxs_by_type = init_run_data['train_dict_embeddings'], init_run_data['dict_idxs_by_type']
                train_dict_indexes = data_process.get_index_from_embeds(train_dict_embeddings, dict_idxs_by_type, force_exact_search=params['force_exact_search'], probe_mult_factor=params['probe_mult_factor'])
                train_men_embeddings, men_idxs_by_type = init_run_data['train_men_embeddings'], init_run_data['men_idxs_by_type']
                train_men_indexes = data_process.get_index_from_embeds(train_men_embeddings, men_idxs_by_type, force_exact_search=params['force_exact_search'], probe_mult_factor=params['probe_mult_factor'])
            else:
                logger.info('Embedding and indexing')
                if dict_embed_data is not None:
                    train_dict_embeddings, train_dict_indexes, dict_idxs_by_type = dict_embed_data['dict_embeds'], dict_embed_data['dict_indexes'], dict_embed_data['dict_idxs_by_type']
                else:
                    train_dict_embeddings, train_dict_indexes, dict_idxs_by_type = data_process.embed_and_index(reranker, entity_dict_vecs, encoder_type="candidate", n_gpu=n_gpu, corpus=entity_dictionary, force_exact_search=params['force_exact_search'], batch_size=params['embed_batch_size'], probe_mult_factor=params['probe_mult_factor'])
                train_men_embeddings, train_men_indexes, men_idxs_by_type = data_process.embed_and_index(reranker, train_men_vecs, encoder_type="context", n_gpu=n_gpu, corpus=train_processed_data, force_exact_search=params['force_exact_search'], batch_size=params['embed_batch_size'], probe_mult_factor=params['probe_mult_factor'])
        else:
            if load_stored_data:
                train_dict_embeddings = init_run_data['train_dict_embeddings']
                train_dict_index = data_process.get_index_from_embeds(train_dict_embeddings, force_exact_search=params['force_exact_search'], probe_mult_factor=params['probe_mult_factor'])
                train_men_embeddings = init_run_data['train_men_embeddings']
                train_men_index = data_process.get_index_from_embeds(train_men_embeddings, force_exact_search=params['force_exact_search'], probe_mult_factor=params['probe_mult_factor'])
            else:
                logger.info('Embedding and indexing')
                if dict_embed_data is not None:
                    train_dict_embeddings, train_dict_index = dict_embed_data['dict_embeds'], dict_embed_data['dict_index']
                else:
                    train_dict_embeddings, train_dict_index = data_process.embed_and_index(reranker, entity_dict_vecs, encoder_type="candidate", n_gpu=n_gpu, force_exact_search=params['force_exact_search'], batch_size=params['embed_batch_size'], probe_mult_factor=params['probe_mult_factor'])
                train_men_embeddings, train_men_index = data_process.embed_and_index(reranker, train_men_vecs, encoder_type="context", n_gpu=n_gpu, force_exact_search=params['force_exact_search'], batch_size=params['embed_batch_size'], probe_mult_factor=params['probe_mult_factor'])

        # Save the initial embeddings and index if this is the first run and data isn't persistent
        if init_base_model_run and not load_stored_data:
            init_run_data = {}
            init_run_data['train_dict_embeddings'] = train_dict_embeddings
            init_run_data['train_men_embeddings'] = train_men_embeddings
            if use_types:
                init_run_data['dict_idxs_by_type'] = dict_idxs_by_type
                init_run_data['men_idxs_by_type'] = men_idxs_by_type
            # NOTE: Cannot pickle faiss index because it is a SwigPyObject
            torch.save(init_run_data, init_run_pkl_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

        init_base_model_run = False

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        logger.info("Starting KNN search...")
        if not use_types:
            _, dict_nns = train_dict_index.search(train_men_embeddings, knn)
        else:
            dict_nns = np.zeros((len(train_men_embeddings), knn))
            for entity_type in train_men_indexes:
                men_embeds_by_type = train_men_embeddings[men_idxs_by_type[entity_type]]
                _, dict_nns_by_type = train_dict_indexes[entity_type].search(men_embeds_by_type, knn)
                dict_nns_idxs = np.array(list(map(lambda x: dict_idxs_by_type[entity_type][x], dict_nns_by_type)))
                for i,idx in enumerate(men_idxs_by_type[entity_type]):
                    dict_nns[idx] = dict_nns_idxs[i]
        logger.info("Search finished")

        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context_inputs, candidate_idxs, n_gold, mention_idxs = batch
            mention_embeddings = train_men_embeddings[mention_idxs.cpu()]
            
            # context_inputs: Shape: batch x token_len
            candidate_inputs = np.array([], dtype=np.int) # Shape: (batch*knn) x token_len
            label_inputs = torch.tensor([[1]+[0]*(knn-1)]*n_gold.sum(), dtype=torch.float32) # Shape: batch(with split rows) x knn
            context_inputs_split = torch.zeros((label_inputs.size(0), context_inputs.size(1)), dtype=torch.long) # Shape: batch(with split rows) x token_len
            # label_inputs = (candidate_idxs >= 0).type(torch.float32) # Shape: batch x knn

            for i, m_embed in enumerate(mention_embeddings):
                knn_dict_idxs = dict_nns[mention_idxs[i]]
                knn_dict_idxs = knn_dict_idxs.astype(np.int64).flatten()
                gold_idxs = candidate_idxs[i][:n_gold[i]].cpu()
                for ng, gold_idx in enumerate(gold_idxs):
                    context_inputs_split[i+ng] = context_inputs[i]
                    candidate_inputs = np.concatenate((candidate_inputs, np.concatenate(([gold_idx], knn_dict_idxs[~np.isin(knn_dict_idxs, gold_idxs)]))[:knn]))
            candidate_inputs = torch.tensor(list(map(lambda x: entity_dict_vecs[x].numpy(), candidate_inputs))).cuda()
            context_inputs_split = context_inputs_split.cuda()
            label_inputs = label_inputs.cuda()
            
            loss, _ = reranker(context_inputs_split, candidate_inputs, label_inputs, pos_neg_loss=params["pos_neg_loss"])

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (params["print_interval"] * grad_acc_steps),
                    )
                )
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                logger.info("Evaluation on the development dataset")
                evaluate(
                    reranker, entity_dict_vecs, valid_men_vecs, device=device, logger=logger, knn=knn, n_gpu=n_gpu, entity_data=entity_dictionary, query_data=valid_processed_data, silent=params["silent"], use_types=use_types or params["use_types_for_eval"], embed_batch_size=params['embed_batch_size'], force_exact_search=use_types or params["use_types_for_eval"] or params["force_exact_search"], probe_mult_factor=params['probe_mult_factor']
                )
                model.train()
                logger.info("\n")

        logger.info("***** Saving fine-tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)
        logger.info(f"Model saved at {epoch_output_folder_path}")

        normalized_accuracy, dict_embed_data = evaluate(
            reranker, entity_dict_vecs, valid_men_vecs, device=device, logger=logger, knn=knn, n_gpu=n_gpu, entity_data=entity_dictionary, query_data=valid_processed_data, silent=params["silent"], use_types=use_types or params["use_types_for_eval"], embed_batch_size=params['embed_batch_size'], force_exact_search=use_types or params["use_types_for_eval"] or params["force_exact_search"], probe_mult_factor=params['probe_mult_factor']
        )

        ls = [best_score, normalized_accuracy]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["path_to_model"] = os.path.join(
        model_output_path, "epoch_{}".format(best_epoch_idx)
    )
    utils.save_model(reranker.model, tokenizer, model_output_path)
    logger.info(f"Best model saved at {model_output_path}")


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    args = parser.parse_args()
    print(args)
    main(args.__dict__)
