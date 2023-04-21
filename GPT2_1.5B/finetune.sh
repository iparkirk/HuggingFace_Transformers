#!/bin/bash

# GPT2_1.5B
# GPT2 fine-tuning using run_clm.py (Causal Language Model)

# CAN'T FIT MODEL 
# torch.cuda.OutOfMemoryError: CUDA out of memory. 
# Tried to allocate 100.00 MiB 
# (GPU 0; 11.78 GiB total capacity; 
#         10.51 GiB already allocated; 
#         39.69 MiB free; 
#         10.62 GiB reserved in total by PyTorch) 
# If reserved memory is >> allocated memory 
# try setting max_split_size_mb to avoid fragmentation.  
# See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

# EITHER
# specify single GPU 1
#export CUDA_VISIBLE_DEVICES=1
##############################

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

