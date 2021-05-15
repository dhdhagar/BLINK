# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Provide an argument parser and default command line options for using BLINK.
import argparse
import importlib
import os
import sys
import datetime


ENT_START_TAG = "[unused1]"
ENT_END_TAG = "[unused2]"
ENT_TITLE_TAG = "[unused3]"


class BlinkParser(argparse.ArgumentParser):
    """
    Provide an opt-producer and CLI arguement parser.

    More options can be added specific by paassing this object and calling
    ''add_arg()'' or add_argument'' on it.

    :param add_blink_args:
        (default True) initializes the default arguments for BLINK package.
    :param add_model_args:
        (default False) initializes the default arguments for loading models,
        including initializing arguments from the model.
    """

    def __init__(
        self, add_blink_args=True, add_model_args=False, 
        description='BLINK parser',
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler='resolve',
            formatter_class=argparse.HelpFormatter,
            add_help=add_blink_args,
        )
        self.blink_home = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        os.environ['BLINK_HOME'] = self.blink_home

        self.add_arg = self.add_argument

        self.overridable = {}

        if add_blink_args:
            self.add_blink_args()
        if add_model_args:
            self.add_model_args()

    def add_blink_args(self, args=None):
        """
        Add common BLINK args across all scripts.
        """
        parser = self.add_argument_group("Common Arguments")
        parser.add_argument(
            "--silent", action="store_true", help="Whether to print progress bars."
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Whether to run in debug mode with only 200 samples.",
        )
        parser.add_argument(
            "--data_parallel",
            action="store_true",
            help="Whether to distributed the candidate generation process.",
        )
        parser.add_argument(
            "--no_cuda", action="store_true", 
            help="Whether not to use CUDA when available",
        )
        parser.add_argument("--top_k", default=10, type=int) 
        parser.add_argument(
            "--seed", type=int, default=52313, help="random seed for initialization"
        )
        parser.add_argument(
            "--zeshel",
            action="store_true",
            help="Whether the dataset is from zeroshot.",
        )

    def add_model_args(self, args=None):
        """
        Add model args.
        """
        parser = self.add_argument_group("Model Arguments")
        parser.add_argument(
            "--max_seq_length",
            default=256,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--max_context_length",
            default=128,
            type=int,
            help="The maximum total context input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--max_cand_length",
            default=128,
            type=int,
            help="The maximum total label input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        ) 
        parser.add_argument(
            "--path_to_model",
            default=None,
            type=str,
            required=False,
            help="The full path to the model to load.",
        )
        parser.add_argument(
            "--bert_model",
            default="bert-base-uncased",
            type=str,
            help="Bert pre-trained model selected in the list: bert-base-uncased, "
            "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
        )
        parser.add_argument(
            "--pull_from_layer", type=int, default=-1, help="Layers to pull from BERT",
        )
        parser.add_argument(
            "--lowercase",
            action="store_true",
            help="Whether to lower case the input text. True for uncased models, False for cased models.",
        )
        parser.add_argument("--context_key", default="context", type=str)
        parser.add_argument(
            "--out_dim", type=int, default=1, help="Output dimention of bi-encoders.",
        )
        parser.add_argument(
            "--add_linear",
            action="store_true",
            help="Whether to add an additonal linear projection on top of BERT.",
        )
        parser.add_argument(
            "--data_path",
            default="data/zeshel",
            type=str,
            help="The path to the train data.",
        )
        parser.add_argument(
            "--output_path",
            default=None,
            type=str,
            required=True,
            help="The output directory where generated output file (model, etc.) is to be dumped.",
        )
        parser.add_argument(
            "--pickle_src_path",
            default=None,
            type=str,
            help="The directory from which to load intermediate processed data to skip redundant computation.",
        )
        parser.add_argument(
            "--embed_batch_size", default=768, type=int, 
            help="Batch size per GPU to use for the embed_and_index method"
        )
        parser.add_argument(
            "--probe_mult_factor", default=1, type=int, 
            help="Mutliplication factor to the square root of the total search space used in FAISS.GpuIndexIVFFlat.nprobe to indicate the number of vectors to compare during search"
        )


    def add_training_args(self, args=None):
        """
        Add model training args.
        """
        parser = self.add_argument_group("Model Training Arguments")
        parser.add_argument(
            "--evaluate", action="store_true", help="Whether to run evaluation."
        )
        parser.add_argument(
            "--only_evaluate", action="store_true", help="Whether to only run eval on the validation set."
        )
        parser.add_argument(
            "--output_eval_file",
            default=None,
            type=str,
            help="The txt file where the the evaluation results will be written.",
        )
        parser.add_argument(
            "--train_batch_size", default=8, type=int, 
            help="Total batch size for training."
        )
        parser.add_argument(
            "--eval_batch_size", default=8, type=int,
            help="Total batch size for evaluation.",
        )
        parser.add_argument(
            "--example_bundle_size", default=32, type=int,
            help="Total batch size for evaluation.",
        )
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument(
            "--learning_rate",
            default=3e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--num_train_epochs",
            default=1,
            type=int,
            help="Number of training epochs.",
        )
        parser.add_argument(
            "--print_interval", type=int, default=5, 
            help="Interval of loss printing",
        )
        parser.add_argument(
           "--eval_interval",
            type=int,
            default=40,
            help="Interval for evaluation during training",
        )
        parser.add_argument(
            "--save_interval", type=int, default=1, 
            help="Interval for model saving"
        )
        parser.add_argument(
            "--warmup_proportion",
            default=0.1,
            type=float,
            help="Proportion of training to perform linear learning rate warmup for. "
            "E.g., 0.1 = 10% of training.",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumualte before performing a backward/update pass.",
        )
        parser.add_argument(
            "--type_optimization",
            type=str,
            default="all_encoder_layers",
            help="Which type of layers to optimize in BERT",
        )
        parser.add_argument(
            "--shuffle", type=bool, default=False, 
            help="Whether to shuffle train data",
        )
        # Cluster-linking arguments
        parser.add_argument(
            "--knn", type=int, default=10, 
            help="Number of kNN (positive+negative) candidates to fetch per mention query during training",
        )
        parser.add_argument(
            "--filter_unlabeled", action="store_true",
            help="Whether to filter mentions that have no labeled entities from the train set",
        )
        parser.add_argument(
            "--use_types", action="store_true",
            help="Whether to pick candidates from only the entities belonging to the mention type",
        )
        parser.add_argument(
            "--pos_neg_loss", action="store_true",
            help="Whether to use both the positive and negative softmax values to compute the loss or to use only the positive",
        )
        parser.add_argument(
            "--force_exact_search", action="store_true",
            help="Whether to run FAISS nearest-neighbour retrieval in exact-search (IndexFlatIP) mode",
        )
        parser.add_argument(
            "--use_types_for_eval", action="store_true",
            help="Whether to use type information during evaluation when --use_types is False",
        )

    def add_eval_args(self, args=None):
        """
        Add model evaluation args.
        """
        parser = self.add_argument_group("Model Evaluation Arguments")
        parser.add_argument(
            "--mode",
            default="valid",
            type=str,
            help="train / valid / test",
        )
        parser.add_argument(
            "--save_topk_result",
            action="store_true",
            help="Whether to save prediction results.",
        )
        parser.add_argument(
            "--encode_batch_size", 
            default=8, 
            type=int, 
            help="Batch size for encoding."
        )
        parser.add_argument(
            "--entity_dict_path",
            default=None,
            type=str,
            help="Path for entity dict when not zeshel",
        )
        parser.add_argument(
            "--cand_pool_path",
            default=None,
            type=str,
            help="Path for candidate pool",
        )
        parser.add_argument(
            "--cand_encode_path",
            default=None,
            type=str,
            help="Path for candidate encoding",
        )
        # Cluster-linking arguments
        parser.add_argument(
            "--graph_mode", type=str, default=None,
            help="Whether to run evaluation in 'directed' or 'undirected' mode. Run both if not specified",
        )
        parser.add_argument(
            "--filter_unlabeled", action="store_true",
            help="Whether to filter mentions that have no labeled entities from the test set",
        )
        parser.add_argument(
            "--knn", type=int, default=10,
            help="Number of kNN mention candidates to fetch per mention query during inference",
        )
        parser.add_argument(
            "--data_split", type=str, default="test",
            help="The split of the dataset to run evaluation on",
        )
        parser.add_argument(
            "--use_types", action="store_true",
            help="Whether to pick candidates from only the entities belonging to the mention type",
        )
        parser.add_argument(
            "--recall_k", type=int, default=16,
            help="Number of kNN entity candidates to fetch to calculate the model's recall accuracy",
        )
        parser.add_argument(
            "--only_recall", action="store_true",
            help="Whether to run evaluation to only compute the recall metric for recall@{--recall_k}",
        )
        parser.add_argument(
            "--force_exact_search", action="store_true",
            help="Whether to run FAISS nearest-neighbour retrieval in exact-search (IndexFlatIP) mode",
        )
        # Entity discovery
        parser.add_argument(
            "--n_thresholds", type=int, default=10,
            help="Number of thresholds to try out for entity discovery",
        )
        parser.add_argument(
            "--exact_threshold", type=float, default=None,
            help="Exact value of the similarity threshold to run the experiment against",
        )
        parser.add_argument(
            "--exact_knn", type=int, default=None,
            help="Exact value of the knn graph to run the experient against",
        )
        parser.add_argument(
            "--embed_data_path",
            default=None,
            type=str,
            help="The directory from which to load the embeddings data (embed_data.t7).",
        )
        parser.add_argument(
            "--drop_all_entities", action="store_true",
            help="Whether to run the discovery without any entities (usually for baseline)",
        )

    def add_joint_train_args(self, args=None):
        """
        Add joint cross train args.
        """
        parser = self.add_argument_group("Joint Model Training Arguments")
        parser.add_argument(
            "--add_sigmoid",
            action="store_true",
            help="Whether to output sigmoid projection of score.",
        )
        parser.add_argument(
            "--objective",
            default="max_margin",
            type=str,
            help="max_margin / softmax",
        )
        parser.add_argument(
            "--margin", 
            default=0.7, 
            type=float, 
            help="margin for triplet max-margin objective"
        )
        parser.add_argument(
            "--pool_highlighted",
            action="store_true",
            help="Whether to score ctxt pairs using highlighted output layers.",
        )

    def add_joint_eval_args(self, args=None):
        """
        Add joint cross evaluation args.
        """
        parser = self.add_argument_group("Joint Model Evaluation Arguments")
        parser.add_argument(
            "--path_to_ctxt_model",
            default=None,
            type=str,
            required=False,
            help="The full path to the ctxt model to load.",
        )
        parser.add_argument(
            "--path_to_cand_model",
            default=None,
            type=str,
            required=False,
            help="The full path to the ctxt model to load.",
        )

