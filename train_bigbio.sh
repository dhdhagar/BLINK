#!/bin/bash

DATASET=$1
DEVICES=$2

# Training
CUDA_VISIBLE_DEVICES=$DEVICES PYTHONPATH="." python blink/biencoder/train_biencoder_mst.py --bert_model=models/biobert-base-cased-v1.1 --data_path=data/$DATASET --output_path=models/trained/$DATASET --pickle_src_path=models/trained/$DATASET/pickle_data --num_train_epochs=5 --train_batch_size=64 --gradient_accumulation_steps=8 --eval_interval=10000 --pos_neg_loss --force_exact_search --embed_batch_size=3500
# CUDA_VISIBLE_DEVICES=$DEVICES PYTHONPATH="." python blink/crossencoder/eval_cluster_linking.py --data_path=data/$DATASET --output_path=models/trained/$DATASET/candidates/arbo --pickle_src_path=models/trained/$DATASET/pickle_data --path_to_biencoder_model=models/trained/$DATASET/pytorch_model.bin --bert_model=models/biobert-base-cased-v1.1 --data_parallel --scoring_batch_size=64 --save_topk_result
# CUDA_VISIBLE_DEVICES=$DEVICES PYTHONPATH="." python blink/crossencoder/original/train_cross.py --data_path=data/$DATASET --pickle_src_path=models/trained/$DATASET/pickle_data --output_path=models/trained/$DATASET/crossencoder/arbo --bert_model=models/biobert-base-cased-v1.1 --learning_rate=2e-05 --num_train_epochs=5 --train_batch_size=2 --eval_batch_size=2 --biencoder_indices_path=models/trained/$DATASET/candidates/arbo --add_linear --skip_initial_eval --eval_interval=-1 --data_parallel

# # Inference
# CUDA_VISIBLE_DEVICES=$DEVICES PYTHONPATH="." python blink/biencoder/eval_cluster_linking.py --bert_model=models/biobert-base-cased-v1.1 --data_path=data/$DATASET --output_path=models/trained/${DATASET}_mst/ --pickle_src_path=models/trained/$DATASET/pickle_data/eval --path_to_model=models/trained/$DATASET/epoch_4/pytorch_model.bin --recall_k=64 --embed_batch_size=3500 --force_exact_search --data_parallel
# CUDA_VISIBLE_DEVICES=$DEVICES PYTHONPATH="." python blink/crossencoder/original/train_cross.py --data_path=data/$DATASET --pickle_src_path=models/trained/$DATASET/pickle_data --output_path=models/trained/$DATASET/crossencoder/eval/arbo --eval_batch_size=2 --biencoder_indices_path=models/trained/$DATASET/candidates/arbo --add_linear --only_evaluate --data_parallel --bert_model=models/biobert-base-cased-v1.1 --path_to_model=models/trained/$DATASET/crossencoder/arbo/pytorch_model.bin
