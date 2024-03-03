export WANDB_PROJECT=Sailor-Finetune

export WANDB_API_KEY=$YOUR_WANDB_API_KEY
export MODEL_NAME="Sailor_4b_sft_aya_orca_lr1e-5_bs_512"
export WANDB_NAME=$MODEL_NAME
export DATA_DIR="dataset"
export DATA_NAME="aya_orca_sft_data"
export BASE_MODEL="sail/Sailor-4B"


accelerate launch \
    --config_file fsdp_config.yaml \
    src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path ${BASE_MODEL} \
    --finetuning_type full \
    --template sailor \
    --dataset_dir ${DATA_DIR} \
    --dataset ${DATA_NAME} \
    --cutoff_len 4096 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --preprocessing_num_workers 64 \
    --flash_attn \
    --max_steps 5000 \
    --save_steps 500 \
    --warmup_steps 100 \
    --output_dir checkpoints/${MODEL_NAME} \
    --bf16 True \
    --plot_loss True \
    --report_to wandb \
    --overwrite_output_dir
