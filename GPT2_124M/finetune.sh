#!/bin/bash

# GPT2_124M
# GPT2 fine-tuning using run_clm.py (Causal Language Model)

# EITHER
# specify single GPU 1
# run over 1 x 12G GPUs => consumed 8GB over one GPU
#export CUDA_VISIBLE_DEVICES=1
##############################

# OR 
# default - distributed over all available GPUs
# run over 2 x 12G GPUs => consumed 10GB over 2 GPUs
python run_clm.py \
    --model_name_or_path ./model \
    --tokenizer_name ./tokenizer \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --output_dir test-clm --overwrite_output_dir

