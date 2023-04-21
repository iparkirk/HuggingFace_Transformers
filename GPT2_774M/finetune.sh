#!/bin/bash

# GPT2_774M
# GPT2 fine-tuning using run_clm.py (Causal Language Model)

# CAN'T FIT MODEL 
# torch.cuda.OutOfMemoryError: CUDA out of memory. 
# Tried to allocate 20.00 MiB 
# (GPU 0; 11.78 GiB total capacity; 
#          6.76 GiB already allocated; 
#          2.19 MiB free; 
#          6.81 GiB reserved in total by PyTorch) 
# If reserved memory is >> allocated memory 
# try setting max_split_size_mb to avoid fragmentation.  
# See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

# EITHER
# specify single GPU 1
#export CUDA_VISIBLE_DEVICES=1

# OR 
# default - distributed over all available GPUs
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
