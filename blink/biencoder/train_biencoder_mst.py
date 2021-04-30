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
from special_partition.special_partition import cluster_linking_partition

import blink.biencoder.data_process_mult as data_process
import blink.biencoder.eval_cluster_linking as eval_cluster_linking
import blink.candidate_ranking.utils as utils
from blink.biencoder.biencoder import BiEncoderRanker
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser

from IPython import embed


logger = None

def evaluate(reranker, valid_dict_vecs, valid_men_vecs, device, logger, knn, n_gpu, entity_data, query_data, silent=False, use_types=False):
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
        print("Eval: Dictionary: Embedding and building index")
        dict_embeds, dict_indexes, dict_idxs_by_type = data_process.embed_and_index(reranker, valid_dict_vecs, encoder_type="candidate", n_gpu=n_gpu, corpus=entity_data)
        print("Eval: Queries: Embedding and building index")
        men_embeds, men_indexes, men_idxs_by_type = data_process.embed_and_index(reranker, valid_men_vecs, encoder_type="context", n_gpu=n_gpu, corpus=query_data)
    else:
        print("Eval: Dictionary: Embedding and building index")
        dict_embeds, dict_index = data_process.embed_and_index(
            reranker, valid_dict_vecs, 'candidate', n_gpu=n_gpu)
        print("Eval: Queries: Embedding and building index")
        men_embeds, men_index = data_process.embed_and_index(
            reranker, valid_men_vecs, 'context', n_gpu=n_gpu)

    for men_query_idx, men_embed in enumerate(tqdm(men_embeds, total=len(men_embeds), desc="Eval: Fetching k-NN")):
        men_embed = np.expand_dims(men_embed, axis=0)
        
        dict_type_idx_mapping, men_type_idx_mapping = None, None
        if use_types:
            entity_type = query_data[men_query_idx]['type']
            dict_index = dict_indexes[entity_type]
            men_index = men_indexes[entity_type]
            dict_type_idx_mapping = dict_idxs_by_type[entity_type]
            men_type_idx_mapping = men_idxs_by_type[entity_type]
        
        # Fetch nearest entity candidate
        dict_cand_idx, dict_cand_score = eval_cluster_linking.get_query_nn(
            reranker, 1, dict_embeds, dict_index, men_embed, type_idx_mapping=dict_type_idx_mapping)
        # Fetch (k+1) NN mention candidates
        men_cand_idxs, men_cand_scores = eval_cluster_linking.get_query_nn(
            reranker, max_knn + 1, men_embeds, men_index, men_embed, type_idx_mapping=men_type_idx_mapping)
        # Filter candidates to remove mention query and keep only the top k candidates
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
        print(f"\nGraph (k={k}):")
        # Partition graph based on cluster-linking constraints
        partitioned_graph, clusters = eval_cluster_linking.partition_graph(
            joint_graphs[k], n_entities, directed=True, return_clusters=True)
        # Infer predictions from clusters
        result = eval_cluster_linking.analyzeClusters(clusters, entity_data, query_data, k)
        acc = float(result['accuracy'].split(' ')[0])
        max_eval_acc = max(acc, max_eval_acc)
        logger.info(f"Eval accuracy for graph@k={k}: {acc}%")
    logger.info(f"Best eval accuracy: {max_eval_acc}%")
    return max_eval_acc

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

    if params["only_evaluate"]:
        evaluate(
            reranker, entity_dict_vecs, valid_men_vecs, device=device, logger=logger, knn=knn, n_gpu=n_gpu, entity_data=entity_dictionary, query_data=valid_processed_data, silent=params["silent"], use_types=use_types
        )
        exit()

    # Get clusters of mentions that map to a gold entity
    train_gold_clusters = data_process.compute_gold_clusters(train_processed_data)
    n_entities = len(entity_dictionary)
    n_mentions = len(train_processed_data)

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
    
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        model.train()
        torch.cuda.empty_cache()
        tr_loss = 0
        results = None

        # Compute mention and entity embeddings at the start of each epoch
        if use_types:
            train_dict_embeddings, train_dict_indexes, dict_idxs_by_type = data_process.embed_and_index(reranker, entity_dict_vecs, encoder_type="candidate", n_gpu=n_gpu, corpus=entity_dictionary)
            train_men_embeddings, train_men_indexes, men_idxs_by_type = data_process.embed_and_index(reranker, train_men_vecs, encoder_type="context", n_gpu=n_gpu, corpus=train_processed_data)
        else:
            train_dict_embeddings, train_dict_index = data_process.embed_and_index(reranker, entity_dict_vecs, encoder_type="candidate", n_gpu=n_gpu)
            train_men_embeddings, train_men_index = data_process.embed_and_index(reranker, train_men_vecs, encoder_type="context", n_gpu=n_gpu)

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        # Store golden MST links
        gold_links = {}

        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context_inputs, candidate_idxs, n_gold, mention_idxs = batch
            mention_embeddings = train_men_embeddings[mention_idxs.cpu()]
            
            # context_inputs: Shape: batch x token_len
            # candidate_inputs = []
            # candidate_inputs = np.array([], dtype=np.long) # Shape: (batch*knn) x token_len
            # label_inputs = (candidate_idxs >= 0).type(torch.float32) # Shape: batch x knn

            positive_idxs = []
            negative_dict_inputs = []
            negative_men_inputs = []

            for m_embed_idx, m_embed in enumerate(mention_embeddings):
                mention_idx = int(mention_idxs[m_embed_idx])
                gold_idxs = set(train_processed_data[mention_idx]['label_idxs'][:n_gold[m_embed_idx]])
                if mention_idx in gold_links:
                    gold_link_idx = gold_links[mention_idx]
                else:
                    # Run MST on mention clusters of all the gold entities of the current query mention to find its positive edge
                    rows, cols, data, shape = [], [], [], (n_entities+n_mentions, n_entities+n_mentions)
                    gold_cluster_mentions = []
                    for cluster_ent in gold_idxs:
                        cluster_mens = train_gold_clusters[cluster_ent]
                        gold_cluster_mentions += cluster_mens
                        for i in range(len(cluster_mens)):
                            from_node = n_entities + cluster_mens[i]
                            to_node = cluster_ent
                            # Add mention-entity link
                            rows.append(from_node)
                            cols.append(to_node)
                            data.append(train_men_embeddings[from_node - n_entities] @ train_dict_embeddings[to_node])
                            # Add forward and reverse mention-mention links
                            for j in range(i+1, len(cluster_mens)):
                                to_node = n_entities + cluster_mens[j]
                                score = train_men_embeddings[from_node - n_entities] @ train_men_embeddings[to_node - n_entities]
                                rows += [from_node, to_node]
                                cols += [to_node, from_node]
                                data += [score, score]
                    gold_cluster_mentions = set(gold_cluster_mentions)
                    # Find MST with entity constraint
                    rows, cols, data = cluster_linking_partition(np.array(rows), np.array(cols), np.array(data), n_entities, directed=True, silent=True)
                    assert len(gold_cluster_mentions) == len(rows)
                    # Store the computed positive edges for the mentions in the clusters if they have the same gold entities as the query mention
                    for i in range(len(rows)):
                        men_idx = rows[i] - n_entities
                        assert men_idx >= 0
                        add_link = True
                        for l in train_processed_data[men_idx]['label_idxs'][:train_processed_data[men_idx]['n_labels']]:
                            if l not in gold_idxs:
                                add_link = False
                                break
                        if add_link:
                            gold_links[rows[i] - n_entities] = cols[i]
                    gold_link_idx = gold_links[mention_idx]
                    
                # Calculate the number of negative entities and mentions to fetch
                knn_dict = knn//2
                knn_men = knn - knn_dict
                if use_types:
                    entity_type = train_processed_data[mention_idxs[m_embed_idx]]['type']
                    train_dict_index = train_dict_indexes[entity_type]
                    train_men_index = train_men_indexes[entity_type]
                _, knn_dict_idxs = train_dict_index.search(np.expand_dims(m_embed, axis=0), knn_dict + int(n_gold[m_embed_idx]))
                knn_dict_idxs = knn_dict_idxs.astype(np.int64).flatten()
                _, knn_men_idxs = train_men_index.search(np.expand_dims(m_embed, axis=0), knn_men + sum([len(train_gold_clusters[gi]) for gi in gold_idxs]))
                knn_men_idxs = knn_men_idxs.astype(np.int64).flatten()
                if use_types:
                    # Map type-specific indices back to the entire list
                    knn_dict_idxs = np.array(list(map(lambda x: dict_idxs_by_type[entity_type][x], knn_dict_idxs)), dtype=np.int64)
                    knn_men_idxs = np.array(list(map(lambda x: men_idxs_by_type[entity_type][x], knn_men_idxs)), dtype=np.int64)
                # Add the positive example
                positive_idxs.append(gold_link_idx)
                # Add the negative examples
                negative_dict_inputs += list(knn_dict_idxs[~np.isin(knn_dict_idxs, list(gold_idxs))][:knn_dict])
                negative_men_inputs += list(knn_men_idxs[~np.isin(knn_men_idxs, list(gold_cluster_mentions))][:knn_men])
            
            negative_dict_inputs = torch.tensor(list(map(lambda x: entity_dict_vecs[x].numpy(), negative_dict_inputs)))
            negative_men_inputs = torch.tensor(list(map(lambda x: train_men_vecs[x].numpy(), negative_men_inputs)))
            positive_embeds = []
            for pos_idx in positive_idxs:
                if pos_idx < n_entities:
                    pos_embed = reranker.encode_candidate(entity_dict_vecs[pos_idx:pos_idx + 1].cuda(), requires_grad=True)
                else:
                    pos_embed = reranker.encode_context(train_men_vecs[pos_idx - n_entities:pos_idx - n_entities + 1].cuda(), requires_grad=True)
                positive_embeds.append(pos_embed)
            positive_embeds = torch.cat(positive_embeds)
            context_inputs = context_inputs.cuda()
            label_inputs = torch.tensor([[1]+[0]*(knn_dict+knn_men)]*len(context_inputs), dtype=torch.float32).cuda()
            
            loss, _ = reranker(context_inputs, label_input=label_inputs, mst_data={
                'positive_embeds': positive_embeds.cuda(),
                'negative_dict_inputs': negative_dict_inputs.cuda(),
                'negative_men_inputs': negative_men_inputs.cuda()
            }, pos_neg_loss=params["pos_neg_loss"])

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
                    reranker, entity_dict_vecs, valid_men_vecs, device=device, logger=logger, knn=knn, n_gpu=n_gpu, entity_data=entity_dictionary, query_data=valid_processed_data, silent=params["silent"], use_types=use_types
                )
                model.train()
                logger.info("\n")

        logger.info("***** Saving fine-tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)
        logger.info(f"Model saved at {epoch_output_folder_path}")

        eval_accuracy = evaluate(
            reranker, entity_dict_vecs, valid_men_vecs, device=device, logger=logger, knn=knn, n_gpu=n_gpu, entity_data=entity_dictionary, query_data=valid_processed_data, silent=params["silent"], use_types=use_types
        )

        ls = [best_score, eval_accuracy]
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
