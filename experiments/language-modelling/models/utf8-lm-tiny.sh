export WANDB_PROJECT="utf8"
export WANDB_NAME="tiny-lm-fineweb"

# tiny-lm using utf8 contains 3m parameters
# Chinchilla scaling law optimal number of tokens is 20x number of parameters
# So for 3.2m parameters, we want to train on 64m tokens
# But! this was calculated for models with standard embeddings, so assuming 16m parameters,
# instead, we train on (128 batch size * 256 block size * 10,000 steps) = 327m tokens

python run_clm.py \
  --use_bit_embeddings True \
  --output_dir ./output-tiny-lm-fineweb \
  --dataset_name HuggingFaceFW/fineweb \
  --streaming True \
  --dataloader_num_workers 1 \
  --dataloader_prefetch_factor 4 \
  --dataloader_pin_memory True \
  --dataloader_persistent_workers True \
  --do_train True \
  --save_strategy steps \
  --max_steps 100000 \
  --save_steps 1000 \
  --save_total_limit 1 \
  --logging_steps 100 \
  --logging_strategy steps \
  --model_name_or_path sbintuitions/tiny-lm \
  --per_device_train_batch_size 256 \
  --block_size 256 \
  --optim adamw_torch_fused \
  --learning_rate 3e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.01 \
  --weight_decay 0.1 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --gradient_checkpointing True \
  --bf16 True \
  --seed 42 \
  --report_to wandb \
  --include_num_input_tokens_seen True

# hf upload sign/utf8-lm-tiny . .