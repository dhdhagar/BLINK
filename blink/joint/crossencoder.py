# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from tqdm import tqdm
from pytorch_transformers.modeling_utils import CONFIG_NAME, WEIGHTS_NAME

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

from pytorch_transformers.modeling_roberta import (
    RobertaConfig,
    RobertaModel,
)

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.tokenization_roberta import RobertaTokenizer

from blink.common.ranker_base import BertEncoder, get_model_obj
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG

from IPython import embed


def load_crossencoder(params):
    # Init model
    crossencoder = CrossEncoderRanker(params)
    return crossencoder


class CrossEncoderModule(torch.nn.Module):
    def __init__(
        self, params, tokenizer, start_mention_id=None, end_mention_id=None
    ):
        super(CrossEncoderModule, self).__init__()
        model_path = params["bert_model"]
        if params.get("roberta"):
            encoder_model = RobertaModel.from_pretrained(model_path)
        else:
            encoder_model = BertModel.from_pretrained(model_path)
        encoder_model.resize_token_embeddings(len(tokenizer))
        self.pool_highlighted = params["pool_highlighted"]
        self.encoder = BertEncoder(
            encoder_model,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"] and not self.pool_highlighted,
            get_all_outputs=self.pool_highlighted
        )
        self.config = self.encoder.bert_model.config
        self.start_mention_id = start_mention_id
        self.end_mention_id = end_mention_id

        if self.pool_highlighted:
            bert_output_dim = encoder_model.embeddings.word_embeddings.weight.size(1)
            output_dim = params["out_dim"]
            self.additional_linear = nn.Linear(2*bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)

    def forward(
        self, token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
    ):
        if self.pool_highlighted:
            with torch.no_grad():
                index_tensor = torch.arange(token_idx_ctxt.shape[1])
                index_tensor = index_tensor.repeat(token_idx_ctxt.shape[0], 1)
                index_tensor = index_tensor.to(token_idx_ctxt.device)

                start_indices = torch.nonzero(
                    token_idx_ctxt == self.start_mention_id, as_tuple=False
                )
                start_indices = start_indices[:,1:]
                end_indices = torch.nonzero(
                    token_idx_ctxt == self.end_mention_id, as_tuple=False
                )
                end_indices = end_indices[:,1:]

                start_indices_a = start_indices[::2,:]
                end_indices_a = end_indices[::2,:]
                start_indices_b = start_indices[1::2,:]
                end_indices_b = end_indices[1::2,:]

                mask_a = (index_tensor > start_indices_a) & (index_tensor < end_indices_a)
                mask_b = (index_tensor > start_indices_b) & (index_tensor < end_indices_b)
                mask_a = mask_a.unsqueeze(-1).type(torch.float)
                mask_b = mask_b.unsqueeze(-1).type(torch.float)

        embedding_ctxt = self.encoder(token_idx_ctxt, segment_idx_ctxt, mask_ctxt)

        if self.pool_highlighted:
            pooled_output_a = (torch.sum(embedding_ctxt * mask_a, 1)
                             / torch.sum(mask_a, 1))
            pooled_output_b = (torch.sum(embedding_ctxt * mask_b, 1)
                             / torch.sum(mask_b, 1))
            pooled_output = torch.cat((pooled_output_a, pooled_output_b), dim=1)
            embedding_ctxt = self.additional_linear(self.dropout(pooled_output))

        return embedding_ctxt.squeeze(-1)


class CrossEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(CrossEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()

        if params.get("roberta"):
            self.tokenizer = RobertaTokenizer.from_pretrained(params["bert_model"],)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(
                params["bert_model"], do_lower_case=params["lowercase"]
            )

        special_tokens_dict = {
            "additional_special_tokens": [
                ENT_START_TAG,
                ENT_END_TAG,
                ENT_TITLE_TAG,
            ],
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.NULL_IDX = self.tokenizer.pad_token_id
        self.START_TOKEN = self.tokenizer.cls_token
        self.END_TOKEN = self.tokenizer.sep_token
        self.START_MENTION_ID = self.tokenizer.convert_tokens_to_ids(ENT_START_TAG)
        self.END_MENTION_ID = self.tokenizer.convert_tokens_to_ids(ENT_END_TAG)

        # keep some parameters around
        self.add_sigmoid = params["add_sigmoid"]
        self.margin = params["margin"]
        self.objective = params["objective"]
        assert self.objective == "softmax" or self.objective == "max_margin"
        
        # init model
        self.build_model()
        if params["path_to_model"] is not None:
            self.load_model(params["path_to_model"])

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict)

    def save(self, output_dir):
        self.save_model(output_dir)
        self.tokenizer.save_vocabulary(output_dir)

    def build_model(self):
        self.model = CrossEncoderModule(
            self.params,
            self.tokenizer,
            start_mention_id=self.START_MENTION_ID,
            end_mention_id=self.END_MENTION_ID
        )
    
    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model) 
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    def score_candidate(self, text_vecs, context_len):
        # Encode contexts first
        num_cand = text_vecs.size(1)
        text_vecs = text_vecs.view(-1, text_vecs.size(-1))
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX, context_len,
        )
        embedding_ctxt = self.model(token_idx_ctxt, segment_idx_ctxt, mask_ctxt,)
        if self.add_sigmoid:
            embedding_ctxt = torch.sigmoid(embedding_ctxt)
        return embedding_ctxt.view(-1, num_cand)

    def forward(self, input_idx, label_input, context_len):
        scores = self.score_candidate(input_idx, context_len)
        if self.objective == "softmax":
            loss = F.cross_entropy(scores, label_input, reduction="mean")
        else:
            assert self.objective == "max_margin"
            mask = torch.zeros_like(scores).type(torch.bool)
            mask[:, label_input] = True
            pos_scores = scores[mask].unsqueeze_(1)
            neg_scores = scores[~mask].reshape(scores.shape[0], -1)
            loss = torch.mean(F.relu(neg_scores - pos_scores + self.margin))
        return loss, scores


def to_bert_input(token_idx, null_idx, segment_pos):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    if segment_pos > 0:
        segment_idx[:, segment_pos:] = token_idx[:, segment_pos:] > 0

    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    # token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask
