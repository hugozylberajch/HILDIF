# BASIC LIBRBARIES
import os
import re
import time
import torch
import random
import pickle
import pandas as pd
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
from tqdm import tqdm
from glob import glob
from copy import deepcopy
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import transformers
from transformers import TrainingArguments
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, BertPreTrainedModel
from transformers import default_data_collator
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataProcessor,
    set_seed,
)
from transformers import (
    InputExample,
    glue_convert_examples_to_features,
    GlueDataset,
    GlueDataTrainingArguments,
)

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler

import nltk
nltk.download('wordnet')
nltk.download('stopwords')

# SPECIFIC LIBRARIES
import faiss
import gensim
from textaugment import Word2vec
from textaugment import Translate
from textaugment import EDA



from Utils import saliency_map

def look_test_lo(model, dataloader, tokenizer, num_display = 20):
  '''
  Returns list of samples from dataloader with Lexical Overlap heuristic

  Args:
    model: trained text classification model
    dataloader: Dataloader 
    tokenizer: tokenizer of trained model
    num_display: number of displayed samples
  '''


  model.eval()
  counter = 0
  test_samples = []
  for idx, test in enumerate(dataloader):
    
    prem_words = []
    hyp_words = []
    premise = ' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[0]
    hypothesis = ' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[1]
    for word in premise.split():
        if word not in [".", "?", "!"]:
            prem_words.append(word.lower())
    for word in hypothesis.split():
        if word not in [".", "?", "!"]:
            hyp_words.append(word.lower())
    prem_filtered = " ".join(prem_words)
    hyp_filtered = " ".join(hyp_words)
    all_in = True
    for word in hyp_words:
        if word not in prem_words:
            all_in = False
            break
    if all_in :
      sample = {}
      sample['attention_mask']= test['attention_mask'].to(device)
      sample['input_ids']=test['input_ids'].to(device)
      sample['labels']=test['labels'].to(device)
      sample['token_type_ids']=test['token_type_ids'].to(device)
      pred = model(**sample)

      if pred.logits[0].argmax().item()!=test['labels'].item():
        counter+=1
        print('TEST CASE ' + str(counter))
        print('Proposition: ',' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[0])
        print('Hypothesis: ',' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[1] )
        print('Class: ', colored('entailment','green') *(test['labels'].item()==1) + colored('non_entailment','red')*(test['labels'].item()==0))
        text = 'Class: '+'entailment'*(test['labels'].item()==1) + 'non_entailment'*(test['labels'].item()==0)
        print('Predicted Class: ',colored('entailment','green') *(model(**sample).logits[0].argmax().item()==1) + colored('non_entailment','red')*(model(**sample).logits[0].argmax().item()==0))
        text = 'Predicted Class: ' + 'entailment' *(pred.logits[0].argmax().item()==1) + 'non_entailment'*(pred.logits[0].argmax().item()==0)
        test_samples.append(test)
    if counter == num_display :
      break

  return

def look_test_sub(model, dataloader, tokenizer, num_display = 20):
  '''
  Returns list of samples from dataloader with Subsequence heuristic

  Args:
    model: trained text classification model
    dataloader: Dataloader 
    tokenizer: tokenizer of trained model
    num_display: number of displayed samples
  '''

  model.eval()
  counter = 0
  test_samples = []
  for idx, test in enumerate(dataloader):
    
    prem_words = []
    hyp_words = []
    premise = ' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[0]
    hypothesis = ' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[1]
    for word in premise.split():
        if word not in [".", "?", "!"]:
            prem_words.append(word.lower())
    for word in hypothesis.split():
        if word not in [".", "?", "!"]:
            hyp_words.append(word.lower())
    prem_filtered = " ".join(prem_words)
    hyp_filtered = " ".join(hyp_words)
    if hyp_filtered in prem_filtered:
      sample = {}
      sample['attention_mask']= test['attention_mask'].to(device)
      sample['input_ids']=test['input_ids'].to(device)
      sample['labels']=test['labels'].to(device)
      sample['token_type_ids']=test['token_type_ids'].to(device)
      pred = model(**sample)

      if pred.logits[0].argmax().item()!=test['labels'].item():
        counter+=1
        print('TEST CASE ' + str(counter))
        print('Proposition: ',' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[0])
        print('Hypothesis: ',' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[1] )
        print('Class: ', colored('entailment','green') *(test['labels'].item()==1) + colored('non_entailment','red')*(test['labels'].item()==0))
        text = 'Class: '+'entailment'*(test['labels'].item()==1) + 'non_entailment'*(test['labels'].item()==0)
        print('Predicted Class: ',colored('entailment','green') *(model(**sample).logits[0].argmax().item()==1) + colored('non_entailment','red')*(model(**sample).logits[0].argmax().item()==0))
        text = 'Predicted Class: ' + 'entailment' *(pred.logits[0].argmax().item()==1) + 'non_entailment'*(pred.logits[0].argmax().item()==0)
        test_samples.append(test)
    if counter == num_display :
      break

  return

def look_test_con(model, dataloader, tokenizer, num_display = 20):
  '''
  Returns list of samples from dataloader with Constituent heuristic

  Args:
    model: trained text classification model
    dataloader: Dataloader 
    tokenizer: tokenizer of trained model
    num_display: number of displayed samples
  '''

  model.eval()
  counter = 0
  test_samples = []
  for idx, test in enumerate(dataloader):
    
    prem_words = []
    hyp_words = []
    premise = ' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[0]
    hypothesis = ' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[1]
    for word in premise.split():
        if word not in [".", "?", "!"]:
            prem_words.append(word.lower())
    for word in hypothesis.split():
        if word not in [".", "?", "!"]:
            hyp_words.append(word.lower())
    prem_filtered = " ".join(prem_words)
    hyp_filtered = " ".join(hyp_words)
    if hyp_filtered in prem_filtered:
      sample = {}
      sample['attention_mask']= test['attention_mask'].to(device)
      sample['input_ids']=test['input_ids'].to(device)
      sample['labels']=test['labels'].to(device)
      sample['token_type_ids']=test['token_type_ids'].to(device)
      pred = model(**sample)

      if pred.logits[0].argmax().item()!=test['labels'].item():
        counter+=1
        print('TEST CASE ' + str(counter))
        print('Proposition: ',' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[0])
        print('Hypothesis: ',' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[1] )
        print('Class: ', colored('entailment','green') *(test['labels'].item()==1) + colored('non_entailment','red')*(test['labels'].item()==0))
        text = 'Class: '+'entailment'*(test['labels'].item()==1) + 'non_entailment'*(test['labels'].item()==0)
        print('Predicted Class: ',colored('entailment','green') *(model(**sample).logits[0].argmax().item()==1) + colored('non_entailment','red')*(model(**sample).logits[0].argmax().item()==0))
        text = 'Predicted Class: ' + 'entailment' *(pred.logits[0].argmax().item()==1) + 'non_entailment'*(pred.logits[0].argmax().item()==0)
        test_samples.append(test)
    if counter == num_display :
      break

  return
