#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0

export output_dir=DEFINE_OUTPUT_DIR_HERE
export saving_dir=$output_dir/"bert_mnli_finetuning" 
export num_gpus=DEFINE_NUM_GPUS_HERE


#python -m torch.distributed.launch --nproc_per_node=$num_gpus \
python examples/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name mnli \
    --per_device_train_batch_size 16 \
    --weight_decay 1e-3 \
    --learning_rate 1e-5 \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --num_train_epochs 10 \
    --output_dir $saving_dir/model \
    --overwrite_output_dir \
    --logging_steps 20 \
    --logging_dir $saving_dir/log \
    --report_to tensorboard \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --warmup_ratio 0.0 \
    --seed 0 \
    --fp16 
