import sys
sys.path.append("..")
# from models import nBestAlignBart
import torch
import logging
from transformers import (
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
    BertTokenizer,
    BartTokenizer,
    AutoConfig,
    AutoTokenizer,
    MBartForConditionalGeneration
)
from transformers import Trainer

class nBestAlignTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")
        loss = model(
            input_ids = input_ids,
            attention_mask = attention_mask
        ).loss

        return loss

def prepare_model(dataset, from_pretrain = True):
    print(f'dataset:{dataset}')
    if (from_pretrain):
        if (dataset in ['AISHELL1', 'AISHELL2']):
            print(f'bart-base-chinese')
            model = AutoModelForSeq2SeqLM.from_pretrained(f'fnlp/bart-base-chinese')
            tokenizer = BertTokenizer.from_pretrained(f'fnlp/bart-base-chinese')
        elif (dataset in ['TEDLIUM2', 'LibriSpeech']): # english
            model = AutoModelForSeq2SeqLM.from_pretrained(f'facebook/bart-base')
            tokenizer = BartTokenizer.from_pretrained(f'facebook/bart-base')

        elif (dataset in ['CSJ']): # japanese
            model = MBartForConditionalGeneration.from_pretrained('ku-nlp/bart-base-japanese')
            tokenizer = AutoTokenizer.from_pretrained('ku-nlp/bart-base-japanese')
    else:
        print(f'Not From Pretrain')
        if (dataset in ['AISHELL1', 'AISHELL2']):
            print(f'bart-base-chinese')
            pretrain_name = f'fnlp/bart-base-chinese'
            tokenizer = BertTokenizer.from_pretrained(pretrain_name)
        elif (dataset in ['TEDLIUM2', 'LibriSpeech']): # english
            pretrain_name = f'facebook/bart-base'
            tokenizer = BartTokenizer.from_pretrained(f'facebook/bart-base')

        elif (dataset in ['CSJ']): # japanese
            pass
            # tokenizer = BartTokenizer.from_pretrained('ClassCat/gpt2-base-japanese-v2')
            config = AutoConfig.from_pretrained(pretrain_name)
            model = BartForConditionalGeneration(config)

        print(f'config:{config}')

    return model, tokenizer