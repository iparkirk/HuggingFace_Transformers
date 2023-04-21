# HuggingFace Pretrained Large Language Models

## Conda 
conda create -n HF python=3.7
conda activate HF
pip install torch 
pip install git+https://github.com/huggingface/transformers
pip install datasets accelerate evaluate scikit-learn

## Download Models/Tokenizer + FineTuning (using single GPU, multple GPUs)
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

