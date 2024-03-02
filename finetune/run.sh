export WANDB_PROJECT=Sailor-Finetune

export WANDB_API_KEY=$YOUR_WANDB_API_KEY
export MODEL_NAME="Sailor_4b_sft_aya_orca_lr1e-5_bs_512"
export WANDB_NAME=$MODEL_NAME


accelerate launch \
    --config_file fsdp_config.yaml \
    src/train_bash.py \
    --stage pt \
    --do_train True \
    --model_name_or_path sail/Sailor-4B \
    --finetuning_type full \
    --template default \
    --dataset_dir "dataset" \
    --dataset aya_orca_sft_data \
    --cutoff_len 4096 \
    --num_train_epochs 3.0 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --preprocessing_num_workers 64 \
    --flash_attn \
    --save_steps 500 \
    --warmup_steps 100 \
    --output_dir checkpoints/$MODEL_NAME \
    --bf16 True \
    --plot_loss True \
    --report_to wandb \
    --overwrite_output_dir
