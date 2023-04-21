# HuggingFace Pretrained Large Language Models

### Conda Environment Setup
```
conda create -n HF python=3.7
conda activate HF
pip install torch 
pip install git+https://github.com/huggingface/transformers
pip install datasets accelerate evaluate scikit-learn
```

### Download Models/Tokenizer + FineTuning (using single GPU, multple GPUs)
```
HuggingFace_Transformers
├── BERT_110M
│   ├── download_models.py
│   ├── finetune.sh
│   └── run_mlm.py
├── BERT_340M
│   ├── download_models.py
│   ├── finetune.sh
│   └── run_mlm.py
├── GPT2_124M
│   ├── download_models.py
│   ├── finetune.sh
│   ├── gen.py
│   └── run_clm.py
├── GPT2_1.5B
│   ├── download_models.py
│   ├── finetune.sh
│   └── run_clm.py
├── GPT2_355M
│   ├── download_models.py
│   ├── finetune.sh
│   └── run_clm.py
└── GPT2_774M
    ├── download_models.py
    ├── finetune.sh
    └── run_clm.py
```


### BERT fine-tuning using run_mlm.py (Masked Language Model)
```
### Tested on 2 x 12GB memory GPUs (NVIDIA TITAN V)

BERT 110M='bert-base-uncased'
#############################
run over 1 x 12G GPUs => consumed 4 GB over one GPUs
run over 2 x 12G GPUs => consumed 6 GB over 2 GPUs

BERT 340M='bert-large-uncased'
#############################
run over 1 x 12G GPUs => consumed 8.5 GB over one GPUs
run over 2 x 12G GPUs => consumed 13 GB over 2 GPUs
```

### GPT2 fine-tuning using run_clm.py (Causal Language Model)
```
### Tested on 2 x 12GB memory GPUs (NVIDIA TITAN V)

GPT2_124M
#########
run over 1 x 12G GPUs => consumed 8GB over one GPU
run over 2 x 12G GPUs => consumed 10GB over 2 GPUs


GPT2_355M
#########
CAN'T FIT MODEL 
torch.cuda.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 16.00 MiB 
(GPU 0; 11.78 GiB total capacity; 
         8.08 GiB already allocated; 
         12.19 MiB free; 
         8.10 GiB reserved in total by PyTorch) 
If reserved memory is >> allocated memory 
try setting max_split_size_mb to avoid fragmentation.  
See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

GPT2_774M
#########
CAN'T FIT MODEL 
torch.cuda.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 20.00 MiB 
(GPU 0; 11.78 GiB total capacity; 
         6.76 GiB already allocated; 
         2.19 MiB free; 
         6.81 GiB reserved in total by PyTorch) 
If reserved memory is >> allocated memory 
try setting max_split_size_mb to avoid fragmentation.  
See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

GPT2_1.5B
#########
CAN'T FIT MODEL 
torch.cuda.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 100.00 MiB 
(GPU 0; 11.78 GiB total capacity; 
        10.51 GiB already allocated; 
        39.69 MiB free; 
        10.62 GiB reserved in total by PyTorch) 
If reserved memory is >> allocated memory 
try setting max_split_size_mb to avoid fragmentation.  
See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF



```
