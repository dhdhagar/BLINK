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
import math

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from pytorch_transformers.optimization import WarmupLinearSchedule
import faiss
from tqdm import tqdm, trange

import blink.biencoder.data_process_mult as data
import blink.candidate_ranking.utils as utils
from blink.biencoder.biencoder import BiEncoderRanker
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser

from IPython import embed


logger = None

def embed_and_index(model, token_id_vecs, encoder_type, batch_size=768, n_gpu=1, only_embed=False, entity_dictionary=None):
    def build_index(embeds):
        # Build index
        d = embeds.shape[1]
        nembeds = embeds.shape[0]
        if nembeds < 10000:  # if the number of embeddings is small, don't approximate
            index = faiss.IndexFlatIP(d)
            index.add(embeds)
        else:
            # number of quantized cells
            nlist = int(math.floor(math.sqrt(nembeds)))
            # number of the quantized cells to probe
            nprobe = int(math.floor(math.sqrt(nlist)))
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(
                quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
            )
            index.train(embeds)
            index.add(embeds)
            index.nprobe = nprobe
        return index
    
    with torch.no_grad():
        if encoder_type == 'context':
            encoder = model.encode_context
        elif encoder_type == 'candidate':
            encoder = model.encode_candidate
        else:
            raise ValueError("Invalid encoder_type: expected context or candidate")

        # Compute embeddings
        embeds = None
        sampler = SequentialSampler(token_id_vecs)
        dataloader = DataLoader(
            token_id_vecs, sampler=sampler, batch_size=(batch_size * n_gpu)
        )
        iter_ = tqdm(dataloader, desc="Embedding in batches")
        for step, batch in enumerate(iter_):
            batch_embeds = encoder(batch.cuda())
            embeds = batch_embeds if embeds is None else np.concatenate((embeds, batch_embeds), axis=0)

        if only_embed:
            return embeds

        if entity_dictionary is None:
            # When "use_types" is False
            index = build_index(embeds)
            return embeds, index
        
        # Build type-specific search indexes
        search_indexes = {}
        entity_dictionary_idxs = {}
        for i,e in enumerate(entity_dictionary):
            ent_type = e['type']
            if ent_type not in entity_dictionary_idxs:
                entity_dictionary_idxs[ent_type] = []
            entity_dictionary_idxs[ent_type].append(i)
        for ent_type in entity_dictionary_idxs:
            search_indexes[ent_type] = build_index(embeds[entity_dictionary_idxs[ent_type]])
        return embeds, search_indexes, entity_dictionary_idxs

# The evaluate function makes a prediction on a set of knn candidates for every mention
def evaluate(
    reranker, eval_dataloader, valid_dict_vecs, params, device, logger, knn, n_gpu, type_data=None
):
    reranker.model.eval()
    
    use_types = type_data is not None # Should be a dict containing "entity_dictionary" and "mention_data"
    
    # To accomodate the approximate-nature of the knn procedure, retrieve more samples and then filter down
    knn = max(16, 2*knn)
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")
    results = {}
    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    if not use_types:
        valid_dict_embeddings, valid_dict_index = embed_and_index(reranker, valid_dict_vecs, encoder_type="candidate", n_gpu=n_gpu)
    else:
        valid_dict_embeddings, valid_dict_indexes, dict_idxs_by_type = embed_and_index(reranker, valid_dict_vecs, encoder_type="candidate", n_gpu=n_gpu, entity_dictionary=type_data['entity_dictionary'])

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_inputs, candidate_idxs, n_gold, mention_idxs = batch
        
        with torch.no_grad():
            mention_embeddings = reranker.encode_context(context_inputs)
            
            # context_inputs: Shape: batch x token_len
            candidate_inputs = np.array([], dtype=np.long) # Shape: (batch*knn) x token_len
            label_inputs = torch.zeros((context_inputs.shape[0], knn), dtype=torch.float32) # Shape: batch x knn

            for i, m_embed in enumerate(mention_embeddings):
                if use_types:
                    entity_type = type_data['mention_data'][mention_idxs[i]]['type']
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
            
            _, logits = reranker(context_inputs, candidate_inputs, label_inputs)

        logits = logits.detach().cpu().numpy()
        tmp_eval_accuracy = int(torch.sum(label_inputs[np.arange(label_inputs.shape[0]), np.argmax(logits, axis=1)] == 1))
        eval_accuracy += tmp_eval_accuracy
        nb_eval_examples += context_inputs.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    results["normalized_accuracy"] = normalized_eval_accuracy
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


def main(params):
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

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
    entity_dictionary_pkl_path = os.path.join(model_output_path, 'entity_dictionary.pickle')
    if os.path.isfile(entity_dictionary_pkl_path):
        print("Loading stored processed entity dictionary...")
        with open(entity_dictionary_pkl_path, 'rb') as read_handle:
            entity_dictionary = pickle.load(read_handle)
        entity_dictionary_loaded = True
    if not params["only_evaluate"]:
        # Load train data
        train_tensor_data_pkl_path = os.path.join(model_output_path, 'train_tensor_data.pickle')
        if os.path.isfile(train_tensor_data_pkl_path):
            print("Loading stored processed train data...")
            with open(train_tensor_data_pkl_path, 'rb') as read_handle:
                train_tensor_data = pickle.load(read_handle)
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

            processed_mention_data, entity_dictionary, train_tensor_data = data.process_mention_data(
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

        # Store the entity dictionary vectors
        entity_dict_vecs = torch.tensor(list(map(lambda x: x['ids'], entity_dictionary)), dtype=torch.long)
        # Store the query mention vectors
        train_men_vecs = train_tensor_data[:][0]

        if params["shuffle"]:
            train_sampler = RandomSampler(train_tensor_data)
        else:
            train_sampler = SequentialSampler(train_tensor_data)

        train_dataloader = DataLoader(
            train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
        )

    # Load eval data
    valid_tensor_data_pkl_path = os.path.join(model_output_path, 'valid_tensor_data.pickle')
    if os.path.isfile(valid_tensor_data_pkl_path):
        print("Loading stored processed valid data...")
        with open(valid_tensor_data_pkl_path, 'rb') as read_handle:
            valid_tensor_data = pickle.load(read_handle)
    else:
        valid_samples = utils.read_dataset("valid", params["data_path"])
        # Check if dataset has multiple ground-truth labels
        mult_labels = "labels" in valid_samples[0].keys()
        # Filter samples without gold entities
        valid_samples = list(filter(lambda sample: (len(sample["labels"]) > 0) if mult_labels else (sample["label"] is not None), valid_samples))
        logger.info("Read %d valid samples." % len(valid_samples))

        processed_valid_mention_data, _, valid_tensor_data = data.process_mention_data(
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

    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )

    evaluate_type_data = {
        'entity_dictionary': entity_dictionary,
        'mention_data': processed_valid_mention_data
    } if use_types else None

    if params["only_evaluate"]:
        # Get the entity dictionary vectors
        entity_dict_vecs = torch.tensor(list(map(lambda x: x['ids'], entity_dictionary)), dtype=torch.long)
        evaluate(
            reranker, valid_dataloader, entity_dict_vecs, params, device=device, logger=logger, knn=knn, n_gpu=n_gpu, type_data=evaluate_type_data
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
    model.train()
    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)
    best_epoch_idx = -1
    best_score = -1
    num_train_epochs = params["num_train_epochs"]
    
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        torch.cuda.empty_cache()
        tr_loss = 0
        results = None

        # Compute mention and entity embeddings at the start of each epoch
        if use_types:
            train_dict_embeddings, train_dict_indexes, dict_idxs_by_type = embed_and_index(reranker, entity_dict_vecs, encoder_type="candidate", n_gpu=n_gpu, entity_dictionary=entity_dictionary)
        else:
            train_dict_embeddings, train_dict_index = embed_and_index(reranker, entity_dict_vecs, encoder_type="candidate", n_gpu=n_gpu)
        train_men_embeddings = embed_and_index(reranker, train_men_vecs, encoder_type="context", n_gpu=n_gpu, only_embed=True)

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context_inputs, candidate_idxs, n_gold, mention_idxs = batch
            mention_embeddings = train_men_embeddings[mention_idxs.cpu()]
            
            # context_inputs: Shape: batch x token_len
            candidate_inputs = np.array([], dtype=np.long) # Shape: (batch*knn) x token_len
            label_inputs = (candidate_idxs >= 0).type(torch.float32) # Shape: batch x knn

            for i, m_embed in enumerate(mention_embeddings):
                if use_types:
                    entity_type = processed_mention_data[mention_idxs[i]]['type']
                    train_dict_index = train_dict_indexes[entity_type]
                _, knn_dict_idxs = train_dict_index.search(np.expand_dims(m_embed, axis=0), knn)
                knn_dict_idxs = knn_dict_idxs.astype(np.int64).flatten()
                if use_types:
                    # Map type-specific indices to the entire dictionary
                    knn_dict_idxs = list(map(lambda x: dict_idxs_by_type[entity_type][x], knn_dict_idxs))
                gold_idxs = candidate_idxs[i][:n_gold[i]].cpu()
                candidate_inputs = np.concatenate((candidate_inputs, np.concatenate((gold_idxs, knn_dict_idxs[~np.isin(knn_dict_idxs, gold_idxs)]))[:knn]))
            candidate_inputs = torch.tensor(list(map(lambda x: entity_dict_vecs[x].numpy(), candidate_inputs))).cuda()
            context_inputs = context_inputs.cuda()
            label_inputs = label_inputs.cuda()
            
            loss, _ = reranker(context_inputs, candidate_inputs, label_inputs)

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
                    reranker, valid_dataloader, entity_dict_vecs, params, device=device, logger=logger, knn=knn, n_gpu=n_gpu, type_data=evaluate_type_data
                )
                model.train()
                logger.info("\n")

        logger.info("***** Saving fine-tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)

        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate(
            reranker, valid_dataloader, entity_dict_vecs, params, device=device, logger=logger, knn=knn, n_gpu=n_gpu, type_data=evaluate_type_data
        )

        ls = [best_score, results["normalized_accuracy"]]
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


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    args = parser.parse_args()
    print(args)
    main(args.__dict__)
