export CUBLAS_WORKSPACE_CONFIG=":16:8" 
export PYTHONHASHSEED=0

export output_dir=DEFINE_OUTPUT_DIR_HERE
export saving_dir=$output_dir
export num_gpus=DEFINE_NUM_GPUS_HERE
export original_model_dir=DEFINE_ORIGINAL_MODEL_DIR_HERE

# compute the importance
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
    examples/text-classification/run_glue.py \
    --model_name_or_path $original_model_dir \
    --task_name mnli \
    --preprocess_importance \
    --do_eval \
    --max_seq_length 128 \
    --output_dir $saving_dir/model \
    --overwrite_output_dir \
    --logging_steps 20 \
    --logging_dir $saving_dir/log \
    --report_to tensorboard \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --metric_for_best_model accuracy \
    --warmup_ratio 0.0 \
    --seed 0 \
    --weight_decay 0.0 \
    --fp16 

# merge the importance
python merge_importance.py --num_files $num_gpus --task mnli --save_dir $saving_dir