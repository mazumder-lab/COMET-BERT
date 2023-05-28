export CUBLAS_WORKSPACE_CONFIG=":16:8" 
export PYTHONHASHSEED=0

export output_dir=DEFINE_OUTPUT_DIR_HERE
export saving_dir=$output_dir
export num_gpus=DEFINE_NUM_GPUS_HERE
export original_model_dir=DEFINE_ORIGINAL_MODEL_DIR_HERE


python examples/text-classification/run_glue.py \
    --model_name_or_path $original_model_dir/model \
    --task_name mnli \
    --per_device_train_batch_size 16 \
    --weight_decay 1e-3 \
    --learning_rate 1e-5 \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --num_train_epochs 50 \
    --output_dir $saving_dir/model \
    --overwrite_output_dir \
    --logging_steps 20 \
    --logging_dir $saving_dir/log \
    --report_to tensorboard \
    --evaluation_strategy epoch \
    --load_best_model_at_end False \
    --metric_for_best_model accuracy \
    --warmup_ratio 0.0 \
    --seed 0 \
    --ignore_data_skip True \
    --cometbert moe \
    --cometbert_distill 1 \
    --cometbert_expert_num 4 \
    --cometbert_expert_dim 768 \
    --cometbert_expert_dropout 0.1 \
    --cometbert_load_balance 0.1 \
    --cometbert_load_importance importance_files/IMPORTANCE_FILE \
    --cometbert_route_method comet-p \
    --cometbert_share_importance 512 \
    --cometbert_gate_entropy 1 \
    --cometbert_gate_gamma 0.1