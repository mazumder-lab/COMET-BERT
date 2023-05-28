# COMET-BERT
This PyTorch package implements [XXX] TO complete

## Installation
* Create and activate conda environment.
``` 
conda env create -f environment.yml
```
* Install Transformers locally.
```
pip install -e .
```
* *Note:* The code is adapted from [this codebase, MoEBERT](https://github.com/SimiaoZuo/MoEBERT).
Arguments regarding *LoRA* and *adapter* can be safely ignored.

## Instructions
COMET-BERT targets task-specific distillation. Before running any distillation code, a pre-trained BERT model should be fine-tuned on the target task.
Path to the fine-tuned model should be passed to `--model_name_or_path`.

### Usage
This code base supports distillation for both GLUE tasks and question answering tasks.
* For GLUE tasks, see `examples/text-classification/run_glue.py`.
* For question answering tasks, see `examples/question-answering/run_qa.py`.

This code base is intended to be used on 3 different stages:
- 1st stage: Fine-tune a BERT model on the target task. (This step natively supported by the Transformers package.)
- 2nd stage: Compute the importance scores for the fine-tuned model. (See below, code from [MoEBERT](https://github.com/SimiaoZuo/MoEBERT).)
- 3rd stage: Distill the fine-tuned model using the importance scores, fine-tune the distilled model on the target task, with comet bert, or comet-perm bert. (See below.)

### Bert Fine-tuning
* For GLUE tasks, see `examples/text-classification/run_glue.py`.
* For question answering tasks, see `examples/question-answering/run_qa.py`.
* An example for that is provided in `scripts_comet/0_finetune_bert_mnli.sh`

It is also possible to use the fine-tuned BERT model provided by the Transformers package, and the HuggingFace API, and skip that first stage

### Importance Score Computation
* Use `scripts_comet/1_importance.sh` to compute the importance scores, 
  This script adds a `--preprocess_importance` argument, removes the `--do_train` argument (important).
* If multiple GPUs are used to compute the importance scores, a `importance_[rank].pkl` file will be saved for each GPU. 
* The script `merge_importance.py` is called in the sh script to merge these files.

### COMET-BERT Distillation
* For GLUE tasks, see `examples/text-classification/run_glue.py`.
* For question answering tasks, see `examples/question-answering/run_qa.py`.
* An example for that is provided in `scripts_comet/2_comet_finetuning.sh`
  * This script loads a fine-tuned BERT model, and a pre-computed importance file (associated with the fine-tuned model).
* The codebase supports different routing strategies: *comet* and *comet-p*
  * Choices should be passed to `--cometbert_route_method`.
  * *comet* is Soft-Tree based routing,
  * *comet-p* is Soft-Tree based routing with permutation local search.
    
