
import sys
import logging
import datasets
from datasets import load_dataset, load_metric
#logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#datasets.logging.set_verbosity_debug()
#load_dataset('squad')
load_dataset('squad_v2')
#load_metric('squad')
load_metric('squad_v2')

# load f1 and exact matc
quit()
configs = ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "mnli_mismatched", "mnli_matched", "qnli", "rte", "wnli", "ax"]
for task in configs:
    load_dataset("glue", task)
    if task == "ax":
        metric = load_metric("glue", "hans")
    else:
        metric = load_metric("glue", task)