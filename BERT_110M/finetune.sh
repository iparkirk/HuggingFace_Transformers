#!/bin/bash

# BERT 110M='bert-base-uncased'
# BERT fine-tuning using run_mlm.py (Masked Language Model)

# EITHER specify single GPU 1
# run over 1 x 12G GPUs => consumed 4 GB over one GPUs
#export CUDA_VISIBLE_DEVICES=1
##############################


# OR default - distributed all available GPUs
# run over 2 x 12G GPUs => consumed 6 GB over 2 GPUs
python run_mlm.py \
    --model_name_or_path ./model \
    --tokenizer_name ./tokenizer \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --output_dir test-clm --overwrite_output_dir

