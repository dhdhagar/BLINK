# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import os
import torch
import torch.nn.functional as F

from pytorch_transformers.modeling_bert import BertModel  # DD1

from pytorch_transformers.tokenization_bert import BertTokenizer
from transformers import AutoTokenizer, AutoModel

from blink.common.ranker_base import BertEncoder, get_model_obj  # For embedding
from blink.common.optimizer import get_bert_optimizer  # optimizer

from IPython import embed


def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoderModule(torch.nn.Module):  # DD2
    def __init__(self, params):
        """
        Params:
        - params : dict
        Dictionary containing configuration options for the bi-encoder module
        """
        super(BiEncoderModule, self).__init__()
        ctxt_bert = AutoModel.from_pretrained(
            params["bert_model"], return_dict=False
        )  # Could be a path containing config.json and pytorch_model.bin; or could be an id shorthand for a model that is loaded in the library
        cand_bert = AutoModel.from_pretrained(params["bert_model"], return_dict=False)
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params[
                "pull_from_layer"
            ],  # from which layer shall we pull the encoded embedding
            add_linear=params[
                "add_linear"
            ],  # whether an additional linear transformation is applied to the output
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = (
            ctxt_bert.config
        )  # Configuration object contains all settings used to initialize the BERT model (number of layers, hidden units, attention heads, etc)

    def forward(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        token_idx_cands,
        segment_idx_cands,
        mask_cands,
    ):
        """
        Returns a tuple containing two elements:
        embedding_ctxt (Tensor): The ENCODED embeddings for the context input. (None if token_idx_ctxt is None).
        embedding_cands (Tensor): The ENCODED embeddings for the candidate inputs. (None if token_idx_cands is None).
        ------
        Params :
        - token_idx_ctxt : Tensor
        Contains token indices for the context input. These indices correspond to the tokens positions in the BERT vocabulary.
        - segment_idx_ctxt : Tensor
        Tensor of segment indices for the context input, used for distinguishing different segments within the same input
        - mask_ctxt : Tensor
        Attention mask for the context input
        - token_idx_cands : Tensor
        Tensor of token indices for the candidate inputs.
        - segment_idx_cands : Tensor
        Tensor of segment indices for the candidate inputs.
        - mask_cands : Tensor
        Attention mask for the candidate inputs.
        """
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands


class BiEncoderRanker(torch.nn.Module):
    "Goal : Encode pieces of text into vector embeddings and then compare these embeddings to perform ranking"

    def __init__(self, params, shared=None):
        # Initialize an instance of the class with specific configurations, resources, and model components.
        super(BiEncoderRanker, self).__init__()
        self.params = params
        # RR1
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        # )
        # self.n_gpu = torch.cuda.device_count()

        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"

        # init bi-encoder model and stores it in self.model
        self.build_model()

        # model_path = params.get("path_to_biencoder_model", params.get("path_to_model"))
        # if model_path:
        #     self.load_model(model_path)

    def load_model(self, fname, cpu=False):
        """
        Loads pre-trained weights into the model from a file. If cpu is true, forces the model to load onto CPU memory.
        ------
        Params :
        - fname : str
        The file name (or path) where the model's state dictionary is saved.
        """
        if cpu:
            state_dict = torch.load(
                fname, map_location=lambda storage, location: "cpu"
            )  # DD6
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(
            state_dict
        )  # Once the state dictionary is loaded into state_dict, the method updates self.model weights

    def build_model(self):
        """
        Constructs the bi-encoder model and stores it in self.model
        """
        self.model = BiEncoderModule(self.params)

    def save_model(self, output_dir):
        """
        Saves the current model weights and configuration to the specified directory.
        ------
        Params :
        - output_dir : str
        The path to the directory where the model's weights and configuration will be saved.
        If this directory does not exist, it will be created.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)  # DD4
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        """
        Initializes and returns an optimizer for the model
        ------
        Params:
        - optim_states : dict
        Contains the state information from a previously used optimizer (number of update, moments of gradient etc...)
        - saved_optim_type : str
        Contains the type of optimization
        """
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],  # Ex : "Adam", "SGD",
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),  # 16-bit floating-point precision
        )

    def encode_context(self, ctxt, requires_grad=False):  # DD7
        """
        Process and encode contextual input with SELF.MODEL()
        ------
        Params:
        - ctxt : Tensor of dim (batch_size, sequence_length) : element (i,j) = INDICE of the j-ième token of the i-ème sequence in the batch
        Collection of token INDICES representing the context inputs
        - requires_grad : bool
        A flag indicating whether the resulting embeddings should require gradient computation
        """
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(ctxt, self.NULL_IDX)
        embedding_context, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )
        """
        if not requires_grad:
            embedding_context = embedding_context.detach()  # Detach but keep on the same device
        return embedding_context
    
        """
        if requires_grad:
            return embedding_context
        return embedding_context.detach().cpu()

    def encode_candidate(self, cands, requires_grad=False):
        """
        Process and encode candidate input with SELF.MODEL()
        ------
        Params:
        - cands : Tensor of dim (cands_number, sequence_length) : element (i,j) = INDICE of the j-ième token of the i-ème sequence in the list of candidate
        Collection of token INDICES representing the candidate inputs
        - requires_grad : bool
        A flag indicating whether the resulting embeddings should require gradient computation
        """

        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )

        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )  # dim = (cands_number, embedding_size)

        if requires_grad:
            return embedding_cands

        return embedding_cands.detach().cpu()

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_vecs is ignored
    def score_candidate(
        self,
        text_vecs,
        cand_vecs,
        random_negs=True,
        cand_encs=None,  # pre-computed candidate encoding.
    ):
        """
        Designed to calculate the similarity scores between context inputs and candidate inputs.
        It supports both scenarios where candidate encodings are pre-computed and where they need to be computed on-the-fly.
        ------
        Params:
        - text_vecs : Tensor of dim (batch_size, sequence_length)
        Collection of token INDICES representing the context inputs
        - cands : Tensor of dim (n, sequence_length) where n is the number of candidate encodings
        Collection of token INDICES representing the candidate inputs
        - random_negs : bool
        A flag indicating whether to use random negatives for scoring
        - cand_encs : A tensor of dim (batch, embed_size)
        Ready for direct comparison with context embeddings
        ------
        - embedding_ctxt : Tensor of dim (batch_size, embed_size)
        - embedding_cands : Tensor of dim (n, embed_size) and Tensor of dim (batch_size, top_k, embed_size) where top_k is the number of candidates for each context
        """
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
            None,
            None,
            None,  # None = token_idx_cands / segment_idx_cands / mask_cands
        )

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            return embedding_ctxt.mm(
                cand_encs.t()
            )  # DD8 Return the score = weight of edges

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None,
            None,
            None,
            token_idx_cands,
            segment_idx_cands,
            mask_cands,  # None = token_idx_ctxt / segment_idx_ctxt / mask_ctxt
        )
        if embedding_cands.shape[0] != embedding_ctxt.shape[0]:  # DD9 #DD10
            embedding_cands = embedding_cands.view(
                embedding_ctxt.shape[0],
                embedding_cands.shape[0] // embedding_ctxt.shape[0],
                embedding_cands.shape[1],
            )  # batchsize x topk x embed_size

        if random_negs:
            # train on random negatives
            ####### Issue here : .mm() expects both operands to be 2D matrices and embedding_cands is a 3D tensor
            ####### Not used because we always train on hard negatives
            return embedding_ctxt.mm(embedding_cands.t())
        else:
            # train on hard negatives
            embedding_ctxt = embedding_ctxt.unsqueeze(2)  # batchsize x embed_size x 1
            scores = torch.bmm(
                embedding_cands, embedding_ctxt
            )  # DD11   # batchsize x topk x 1    ## Score = weight of edges
            scores = torch.squeeze(
                scores, dim=2
            )  # batchsize x topk   # Removes the dimension at index 2 of the scores tensor
            return scores  # element (i,j) = weight between mention_i and candidate_j

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(
        self,
        context_input,
        cand_input=None,
        label_input=None,
        mst_data=None,
        pos_neg_loss=False,
        only_logits=False,
    ):
        """
        Computes the loss and score of the model for a batch of data

        Parameters
        ----------
        - context_input : Tensor of dim (batch_size, sequence_length)
        Token INDICES representing the context inputs
        - cand_input : Tensor of dim (batch_size, sequence_length)
        Token INDICES representing the candidate inputs
        - label_input : Tensor containing binary values that act as indicator variables
        Contains Indicator variable such that I_{u,m_i} = 1 if(u,mi) ∈ E'_{m_i} and I{u,m_i} = 0 otherwise.
        - mst_data :
        Labels indicating the correct and negative candidates for the given contexts. Used for computing loss during training.
        - pos_neg_loss : bool
        Flag to indicate whether to use a specific loss function that considers both positive and negative samples distinctly.
        - only_digits : bool
        Flag to return only the logits (scores) without computing the loss. Useful for evaluation or inference.
        """
        if mst_data is not None:
            context_embeds = self.encode_context(
                context_input, requires_grad=True
            ).unsqueeze(
                2
            )  # batchsize x embed_size x 1

            pos_embeds = mst_data["positive_embeds"].unsqueeze(
                1
            )  # batchsize x 1 x embed_size # this is POSITIVE EMBEDDING !

            neg_dict_embeds = self.encode_candidate(
                mst_data["negative_dict_inputs"], requires_grad=True
            )  # (batchsize*knn_dict) x embed_size : need reshaping

            neg_dict_embeds = neg_dict_embeds.view(
                context_embeds.shape[0],
                neg_dict_embeds.shape[0] // context_embeds.shape[0],
                neg_dict_embeds.shape[1],
            )  # batchsize x knn_dict x embed_size

            cand_embeds = torch.cat(
                (pos_embeds, neg_dict_embeds), dim=1
            )  # batchsize x knn_dict+1 x embed_size

            if mst_data["negative_men_inputs"] is not None:
                neg_men_embeds = self.encode_context(
                    mst_data["negative_men_inputs"], requires_grad=True
                )  # (batchsize*knn_men) x embed_size

                neg_men_embeds = neg_men_embeds.view(
                    context_embeds.shape[0],
                    neg_men_embeds.shape[0] // context_embeds.shape[0],
                    neg_men_embeds.shape[1],
                )  # batchsize x knn_men x embed_size

                cand_embeds = torch.cat(
                    (cand_embeds, neg_men_embeds), dim=1
                )  # batchsize x (knn_men + knn_ent + 1) x embed_size #DD12

            # Compute scores
            scores = torch.bmm(cand_embeds, context_embeds)  # batchsize x topk x 1
            scores = torch.squeeze(scores, dim=2)  # batchsize x topk

        else:  # Case 2
            flag = label_input is None  # True means no negative examples
            scores = self.score_candidate(context_input, cand_input, flag)
            bs = scores.size(0)  # batch_size

        if only_logits:
            return scores

        if label_input is None:
            target = torch.LongTensor(torch.arange(bs))
            ###### The target is np.array([0,1, ..., bs-1]) which is clearly not the real target.
            ###### This if is probably never used as we are interested in the mst performance.
            loss = F.cross_entropy(
                scores, target, reduction="mean"
            )  # F = torch.nn.functional
        else:
            if pos_neg_loss:  # DD13 # This should also be the one always used.
                loss = torch.mean(
                    torch.sum(
                        -torch.log(torch.softmax(scores, dim=1) + 1e-8) * label_input
                        - torch.log(1 - torch.softmax(scores, dim=1) + 1e-8)
                        * (1 - label_input),
                        dim=1,
                    )
                )
            else:  # Uses a simpler loss calculation focusing only on the negative log likelihood of the positive labels
                # Suitable for more traditional classification tasks where the focus is on maximizing the probability of the correct class.
                loss = torch.mean(
                    torch.max(
                        -torch.log(torch.softmax(scores, dim=1) + 1e-8) * label_input,
                        dim=1,
                    )[0]
                )
        return loss, scores


def to_bert_input(token_idx, null_idx):
    """
    Prepare token_idx, segment_idx and mask for encoding
    ------
    Params :
    - token_idx : Tensor of dim (batch_size, sequence_length)
    Contains INDICES of input tokens.
    - segment_idx : Tensor of dim (batch_size, sequence_length)
    For question answering : A / B
    - null_idx : int
    Represents the index value used to indicate padding.
    - mask : Tensor of bool
    Indicates which tokens should be attended to (True) and which should be ignored (False) during the processing by BERT.
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask
