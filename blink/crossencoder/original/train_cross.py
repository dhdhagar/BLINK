# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import collections
import os
import torch
import random
import time
import numpy as np
import pickle
import json
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from pytorch_transformers.optimization import WarmupLinearSchedule
from blink.crossencoder.original.crossencoder import CrossEncoderRanker
import blink.candidate_ranking.utils as utils
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser
import blink.biencoder.data_process_mult as data_process
from IPython import embed


logger = None


def modify(context_input, candidate_input, max_seq_length):
    new_input = []
    context_input = context_input.tolist()
    candidate_input = candidate_input.tolist()

    for i in tqdm(range(len(context_input)), desc="Concatenating"):
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


def evaluate(reranker, eval_dataloader, device, logger, context_length, silent=True, unfiltered_length=None,
             mention_data=None, compute_macro_avg=False, store_failure_success=False):
    reranker.model.eval()
    if silent:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {

    }

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    if mention_data is not None:
        processed_mention_data = mention_data['mention_data']
        n_mentions_per_type = collections.defaultdict(int)
        if compute_macro_avg:
            for men in processed_mention_data:
                n_mentions_per_type[men["type"]] += 1
        dictionary = mention_data['entity_dictionary']
        stored_candidates = mention_data['stored_candidates']
        if store_failure_success:
            failsucc = {
                'failure': [],
                'success': []
            }
    n_evaluated_per_type = collections.defaultdict(int)
    n_hits_per_type = collections.defaultdict(int)

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, label_input, mention_idxs = batch
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, label_input, context_length)

        logits = logits.detach().cpu().numpy()
        label_ids = label_input.cpu().numpy()

        tmp_eval_hits, predicted = utils.accuracy(logits, label_ids, return_bool_arr=True)
        tmp_eval_accuracy = np.sum(tmp_eval_hits)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += context_input.size(0)
        nb_eval_steps += 1

        if compute_macro_avg:
            for i, m_idx in enumerate(mention_idxs):
                type = processed_mention_data[m_idx]['type']
                n_evaluated_per_type[type] += 1
                is_hit = tmp_eval_hits[i]
                n_hits_per_type[type] += is_hit

        if store_failure_success:
            for i, m_idx in enumerate(mention_idxs):
                men_query = processed_mention_data[m_idx]
                dict_pred = dictionary[stored_candidates['candidates'][m_idx][predicted[i]]]
                report_obj = {
                    'mention_id': men_query['mention_id'],
                    'mention_name': men_query['mention_name'],
                    'mention_gold_cui': '|'.join(men_query['label_cuis']),
                    'mention_gold_cui_name': '|'.join(
                        [dictionary[i]['title'] for i in men_query['label_idxs'][:men_query['n_labels']]]),
                    'predicted_name': dict_pred['title'],
                    'predicted_cui': dict_pred['cui']
                }
                failsucc['success' if tmp_eval_hits[i] else 'failure'].append(report_obj)

    results["filtered_length"] = nb_eval_examples
    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    results["normalized_accuracy"] = normalized_eval_accuracy
    if unfiltered_length is not None:
        results["unfiltered_length"] = unfiltered_length
        results["unnormalized_accuracy"] = eval_accuracy / unfiltered_length
    if compute_macro_avg:
        norm_macro, unnorm_macro = 0, 0
        for type in n_mentions_per_type:
            norm_macro += n_hits_per_type[type]/n_evaluated_per_type[type]
            unnorm_macro += n_hits_per_type[type]/n_mentions_per_type[type]
        norm_macro /= len(n_evaluated_per_type)
        unnorm_macro /= len(n_mentions_per_type)
        results["normalized_macro_avg_acc"] = norm_macro
        results["unnormalized_macro_avg_acc"] = unnorm_macro
    if store_failure_success:
        results['failure'] = failsucc['failure']
        results['success'] = failsucc['success']
    if not store_failure_success:
        logger.info(json.dumps(results))
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

def load_data(data_split,
              bi_tokenizer,
              max_context_length,
              max_cand_length,
              knn,
              pickle_src_path,
              params,
              logger,
              return_dict_only=False):
    entity_dictionary_loaded = False
    entity_dictionary_pkl_path = os.path.join(pickle_src_path, 'entity_dictionary.pickle')
    if os.path.isfile(entity_dictionary_pkl_path):
        print("Loading stored processed entity dictionary...")
        with open(entity_dictionary_pkl_path, 'rb') as read_handle:
            entity_dictionary = pickle.load(read_handle)
        entity_dictionary_loaded = True

    if return_dict_only and entity_dictionary_loaded:
        return entity_dictionary

    # Load data
    tensor_data_pkl_path = os.path.join(pickle_src_path, f'{data_split}_tensor_data.pickle')
    processed_data_pkl_path = os.path.join(pickle_src_path, f'{data_split}_processed_data.pickle')
    if os.path.isfile(tensor_data_pkl_path) and os.path.isfile(processed_data_pkl_path):
        print("Loading stored processed data...")
        with open(tensor_data_pkl_path, 'rb') as read_handle:
            tensor_data = pickle.load(read_handle)
        with open(processed_data_pkl_path, 'rb') as read_handle:
            processed_data = pickle.load(read_handle)
    else:
        data_samples = utils.read_dataset(data_split, params["data_path"])
        if not entity_dictionary_loaded:
            with open(os.path.join(params["data_path"], 'dictionary.pickle'), 'rb') as read_handle:
                entity_dictionary = pickle.load(read_handle)

        # Check if dataset has multiple ground-truth labels
        mult_labels = "labels" in data_samples[0].keys()
        # Filter samples without gold entities
        data_samples = list(
            filter(lambda sample: (len(sample["labels"]) > 0) if mult_labels else (sample["label"] is not None),
                   data_samples))
        logger.info("Read %d data samples." % len(data_samples))

        processed_data, entity_dictionary, tensor_data = data_process.process_mention_data(
            data_samples,
            entity_dictionary,
            bi_tokenizer,
            max_context_length,
            max_cand_length,
            context_key=params["context_key"],
            multi_label_key="labels" if mult_labels else None,
            silent=params["silent"],
            logger=logger,
            debug=params["debug"],
            knn=knn,
            dictionary_processed=entity_dictionary_loaded
        )
        print("Saving processed data...")
        if not entity_dictionary_loaded:
            with open(entity_dictionary_pkl_path, 'wb') as write_handle:
                pickle.dump(entity_dictionary, write_handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        with open(tensor_data_pkl_path, 'wb') as write_handle:
            pickle.dump(tensor_data, write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(processed_data_pkl_path, 'wb') as write_handle:
            pickle.dump(processed_data, write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    if return_dict_only:
        return entity_dictionary
    return entity_dictionary, tensor_data, processed_data


def get_data_loader(data_split, tokenizer, context_length, candidate_length, max_seq_length, pickle_src_path,
                    logger, inject_ground_truth=False, max_n=None, shuffle=True, return_data=False):
    # Load the top-64 indices for each mention query and the ground truth label if it exists in the candidate set
    logger.info(f"Loading {data_split} data...")
    fname = os.path.join(params["biencoder_indices_path"], f"candidates_{data_split}_top64.t7")
    stored_data = torch.load(fname)
    entity_dictionary, tensor_data, processed_data = load_data(data_split,
                                                               tokenizer,
                                                               context_length,
                                                               candidate_length,
                                                               1,
                                                               pickle_src_path,
                                                               params,
                                                               logger)
    logger.info("Loaded")
    dict_vecs = list(map(lambda x: x['ids'], entity_dictionary))

    mention_idxs = torch.tensor([i for i in range(len(stored_data['labels']))])
    candidate_input = []
    keep_mask = [True] * len(stored_data['labels'])
    for i in tqdm(range(len(stored_data['labels'])), desc="Processing"):
        if stored_data['labels'][i] == -1:
            # If ground truth not in candidates, replace the last candidate with the ground truth
            if inject_ground_truth:
                gold_idx = processed_data[i]["label_idxs"][0]
                stored_data['labels'][i] = len(stored_data['candidates'][i]) - 1
                stored_data['candidates'][i][-1] = gold_idx
            else:
                keep_mask[i] = False
                continue
        cands = list(map(lambda x: dict_vecs[x], stored_data['candidates'][i]))
        candidate_input.append(cands)
    candidate_input = np.array(candidate_input)
    context_input = tensor_data[:][0][keep_mask]
    label_input = torch.tensor(stored_data['labels'])[keep_mask]
    mention_idxs = mention_idxs[keep_mask]

    n_no_label = len(stored_data['labels']) - np.sum(keep_mask)

    if max_n is not None:
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]

    if params["debug"]:
        max_n = 200
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]

    context_input = modify(context_input, candidate_input, max_seq_length)
    tensor_data = TensorDataset(context_input, label_input, mention_idxs)
    sampler = RandomSampler(tensor_data) if shuffle else SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data,
        sampler=sampler,
        batch_size=params["train_batch_size" if data_split == 'train' else "eval_batch_size"]
    )
    if return_data:
        return dataloader, n_no_label, {"entity_dictionary": entity_dictionary,
                                        "mention_data": processed_data,
                                        "stored_candidates": stored_data}
    return dataloader, n_no_label


def main(params):
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    pickle_src_path = params["pickle_src_path"]
    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = CrossEncoderRanker(params)
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
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    max_seq_length = params["max_seq_length"]
    context_length = params["max_context_length"]
    candidate_length = params["max_context_length"]

    if params["only_evaluate"]:
        split = "test" if params["custom_cand_set"] is None else params["custom_cand_set"]
        test_dataloader, n_test_skipped, data = get_data_loader(split, tokenizer, context_length, candidate_length,
                                                                max_seq_length, pickle_src_path, logger,
                                                                inject_ground_truth=params["inject_eval_ground_truth"],
                                                                shuffle=False, return_data=True)
        logger.info("Evaluating the model on the test set")
        results = evaluate(
            reranker,
            test_dataloader,
            device=device,
            logger=logger,
            context_length=context_length,
            silent=params["silent"],
            unfiltered_length=len(data["mention_data"]),
            mention_data=data,
            compute_macro_avg=True,
            store_failure_success=True
        )
        results_path = os.path.join(model_output_path, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            print(f"\nAnalysis saved at: {results_path}")
        exit()

    train_dataloader, _, train_data = get_data_loader('train', tokenizer, context_length, candidate_length,
                                                      max_seq_length, pickle_src_path, logger,
                                                      inject_ground_truth=params["inject_train_ground_truth"],
                                                      return_data=True)

    valid_dataloader, n_valid_skipped = get_data_loader('valid', tokenizer, context_length, candidate_length,
                                                        max_seq_length, pickle_src_path, logger,
                                                        inject_ground_truth=params["inject_eval_ground_truth"],
                                                        max_n=2048)

    if not params["skip_initial_eval"]:
        logger.info("Evaluating dev set on untrained model...")
        # Evaluate before training
        results = evaluate(
            reranker,
            valid_dataloader,
            device=device,
            logger=logger,
            context_length=context_length,
            silent=params["silent"],
        )

    time_start = time.time()

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_data["mention_data"]), logger)

    model.train()

    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = params["num_train_epochs"]

    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        results = None

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        part = 0
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context_input, label_input, _ = batch
            loss, _ = reranker(context_input, label_input, context_length)

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

            if params["eval_interval"] != -1:
                if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                    logger.info("Evaluation on the development dataset")
                    evaluate(
                        reranker,
                        valid_dataloader,
                        device=device,
                        logger=logger,
                        context_length=context_length,
                        silent=params["silent"],
                    )
                    logger.info("***** Saving fine - tuned model *****")
                    epoch_output_folder_path = os.path.join(
                        model_output_path, "epoch_{}_{}".format(epoch_idx, part)
                    )
                    part += 1
                    utils.save_model(model, tokenizer, epoch_output_folder_path)
                    model.train()
                    logger.info("\n")

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)

        results = evaluate(
            reranker,
            valid_dataloader,
            device=device,
            logger=logger,
            context_length=context_length,
            silent=params["silent"],
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


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    args = parser.parse_args()
    print(args)
    params = args.__dict__
    main(params)
