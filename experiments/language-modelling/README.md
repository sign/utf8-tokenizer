# Bit-Biased Byte-Level Language Modelling

> [!TIP]
> This trains on a macOS system with M4 chip in 10 minutes~


We modified the `run_clm.py` script, in this directory.
See the modifications by running a diff with: 
https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

- We force the use of `UTF8Tokenizer`.
- We add an argument `--use_bit_embeddings` to use the `patch_embedding_layer`


## Setup

```shell
# install the library and training dependencies.
pip install ".[train]" 

# Login to Weights & Biases
wandb login
export WANDB_PROJECT="clm-bit-embeddings"
```

## Training

We train `sbintuitions/tiny-lm` using:

```shell
python run_clm.py \
    --use_bit_embeddings False \
    --output_dir ./output-clm-byte \
```

Compared to our model setup:
```shell
python run_clm.py \
    --use_bit_embeddings True \
    --output_dir ./output-clm-byte-bit \
```

With the following shared arguments:

```shell
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train True \
    --do_eval True \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 100 \
    --logging_strategy steps \
    --num_train_epochs 1 \
    --model_name_or_path sbintuitions/tiny-lm \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --block_size 512 \
    --optim adamw_torch_fused \
    --bf16 True \
    --seed 42 \
    --report_to wandb \
    --include_num_input_tokens_seen True
```

If you want to use a different tokenizer, you can specify it with:
```shell
    --tokenizer_name "google/byt5-small"
```