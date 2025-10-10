export WANDB_PROJECT="clm-bit-embeddings"
export WANDB_NAME="clm-byte-t5"
export WANDB_RUN_GROUP="byte-t5"

python run_clm.py \
    --output_dir ./output-clm-byte-t5 \
    --tokenizer_name "google/byt5-small" \
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
