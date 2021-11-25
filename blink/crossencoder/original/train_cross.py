# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import torch
import random
import time
import numpy as np
import pickle
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


def evaluate(reranker, eval_dataloader, device, logger, context_length, silent=True):
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
    logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
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


def main(params):
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = CrossEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    # utils.save_model(model, tokenizer, model_output_path)

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

    pickle_src_path = params["pickle_src_path"]

    fname = os.path.join(params["biencoder_indices_path"], "candidates_train_top64.t7") # train.t7
    train_data = torch.load(fname) # Contains the top-64 indices for each mention query and the ground truth label if it exists in the candidate set
    entity_dictionary, tensor_data, processed_data = load_data('train',
                                                               tokenizer,
                                                               context_length,
                                                               candidate_length,
                                                               1,
                                                               pickle_src_path,
                                                               params,
                                                               logger)
    dict_vecs = torch.tensor(list(map(lambda x: x['ids'], entity_dictionary)), dtype=torch.long)

    # If ground truth not in candidates, replace the last candidate with the ground truth
    candidate_input = []
    for i in range(len(train_data['labels'])):
        if train_data['labels'][i] == -1:
            gold_idx = processed_data[i]["label_idxs"][0]
            train_data['labels'][i] = len(train_data['candidates'][i]) - 1
            train_data['candidates'][i][-1] = gold_idx
        cands = list(map(lambda x: dict_vecs[x], train_data['candidates'][i]))
        candidate_input.append(cands)
    context_input = tensor_data[:][0]
    label_input = train_data['labels']

    # train_data = torch.load(fname)
    # context_input = train_data["context_vecs"]
    # candidate_input = train_data["candidate_vecs"]
    # label_input = train_data["labels"]

    if params["debug"]:
        max_n = 200
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]

    context_input = modify(context_input, candidate_input, max_seq_length)

    train_tensor_data = TensorDataset(context_input, label_input)
    train_sampler = RandomSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, 
        sampler=train_sampler, 
        batch_size=params["train_batch_size"]
    )

    max_n = 2048
    if params["debug"]:
        max_n = 200
    fname = os.path.join(params["biencoder_indices_path"], "candidates_valid_top64.t7")
    valid_data = torch.load(fname)

    _, tensor_data, processed_data = load_data('valid',
                                               tokenizer,
                                               context_length,
                                               candidate_length,
                                               1,
                                               pickle_src_path,
                                               params,
                                               logger)
    candidate_input = []
    keep_mask = [True]*len(valid_data['labels'])
    for i in range(len(valid_data['labels'])):
        if valid_data['labels'][i] == -1:
            keep_mask[i] = False
            continue
        cands = list(map(lambda x: dict_vecs[x], valid_data['candidates'][i]))
        candidate_input.append(cands)
    candidate_input = np.array(candidate_input)
    context_input = tensor_data[:][0][keep_mask]
    label_input = np.array(valid_data["labels"])[keep_mask]

    context_input = context_input[:max_n]
    candidate_input = candidate_input[:max_n]
    label_input = label_input[:max_n]

    context_input = modify(context_input, candidate_input, max_seq_length)

    valid_tensor_data = TensorDataset(context_input, label_input)
    valid_sampler = SequentialSampler(valid_tensor_data)

    valid_dataloader = DataLoader(
        valid_tensor_data, 
        sampler=valid_sampler, 
        batch_size=params["eval_batch_size"]
    )

    # evaluate before training
    results = evaluate(
        reranker,
        valid_dataloader,
        device=device,
        logger=logger,
        context_length=context_length,
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

    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)

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
            context_input, label_input = batch
            loss, _ = reranker(context_input, label_input, context_length)

            # if n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu.

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
        # reranker.save(epoch_output_folder_path)

        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
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
    params["path_to_model"] = os.path.join(
        model_output_path, "epoch_{}".format(best_epoch_idx)
    )
    # utils.save_model(reranker.model, tokenizer, model_output_path)
    # reranker = utils.get_biencoder(params)
    # reranker.save(model_output_path)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
