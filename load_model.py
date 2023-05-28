import argparse
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='distilbert-base-uncased')
args = parser.parse_args()
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
# transformers.logging.set_verbosity_debug()
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(args.model)

model = 'deepset/bert-base-uncased-squad2'
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(model)