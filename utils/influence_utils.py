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

# Influence function
num_synonym = 50

# Setting random seed and device
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

s_test_damp, s_test_scale, s_test_num_samples = influence_helpers.select_s_test_config(
        trained_on_task_name="mnli-2",
        train_task_name="mnli-2",
        eval_task_name="mnli-2")


def get_test(model, dataloader, tokenizer, num_display = 20, num_test_sample = 10, get_all = False, test_case = None):
  '''
  Return array of anchor points from the dataloader. Can be random or handpicked. 

  Args: 
    model: trained text classification model
    dataloader: Validation Dataloader 
    tokenizer: tokenizer of trained model
    num_display: number of displayed samples
    num_test_sample: number of picked anchor points
    get_all: if True, all the anchor points are picked, if False, only selected one from user
    test_case: name of the test case for log
  '''


  if test_case is not None : 
    log = open(r'./ISO/log/' +test_case + '.txt', "w")

  model.eval()
  counter = 0
  test_samples = []
  for idx, test in enumerate(dataloader):
    sample = {}
    sample['attention_mask']= test['attention_mask'].to(device)
    sample['input_ids']=test['input_ids'].to(device)
    sample['labels']=test['labels'].to(device)
    sample['token_type_ids']=test['token_type_ids'].to(device)
    pred = model(**sample)
    if pred.logits[0].argmax().item() != test['labels'].item() :
      counter+=1
      print('TEST CASE ' + str(counter))
      log.write('TEST CASE ' + str(counter) +'\n')
      print('Proposition: ',' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[0])
      log.write('Proposition: '+' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[0]+'\n')
      print('Hypothesis: ',' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[1] )
      log.write('Hypothesis: '+' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[1]+'\n')
      print('Class: ', colored('entailment','green') *(test['labels'].item()==1) + colored('non_entailment','red')*(test['labels'].item()==0))
      text = 'Class: '+'entailment'*(test['labels'].item()==1) + 'non_entailment'*(test['labels'].item()==0)
      log.write(text+'\n')
      print('Predicted Class: ',colored('entailment','green') *(model(**sample).logits[0].argmax().item()==1) + colored('non_entailment','red')*(model(**sample).logits[0].argmax().item()==0))
      text = 'Predicted Class: ' + 'entailment' *(model(**sample).logits[0].argmax().item()==1) + 'non_entailment'*(model(**sample).logits[0].argmax().item()==0)
      log.write(text+'\n')
      test_samples.append(test)
    if counter == num_test_sample :
      break
  counter2 = 0
  if get_all : 
    indices_to_keep = np.arange(num_test_sample)
  else : 
    while counter2 < num_test_sample:
      text = 'Test Sample to keep number '+str(counter2)+': '
      idx = int(input(text))
      indices_to_keep.append(idx-1)
      counter2+=1

  return np.array(test_samples)[indices_to_keep]

def get_influence(model, dataloader, test_samples, num_test_sample = 10, num_influential_sample = 5, k=3000):
  '''
  Return array of anchor points, influential samples for each anchor points and influence score for each influential sample. 

  Args: 
    model: trained text classification model
    dataloader: Training Dataloader 
    test_samples: array of anchor points
    num_test_sample: number of picked anchor points
    num_influential_sample: number of influential sample for each anchor point
    k: KNN parameter for faiss
  '''

  model.eval()
  counter = 0
  influence_scores = []
  influence_samples = []

  # FIND KNN
  for test in test_samples:
      sample = {}
      sample['attention_mask']= test['attention_mask'].to(device)
      sample['input_ids']=test['input_ids'].to(device)
      sample['labels']=test['labels'].to(device)
      sample['token_type_ids']=test['token_type_ids'].to(device)
      features = misc_utils.compute_BERT_CLS_feature(model, **sample)
      features = features.cpu().detach().numpy()
      KNN_distances, KNN_indices = faiss_index_mnli.search(
                k=k, queries=features)
      
      # GET INFLUENTIAL TRAINING
      influences, index, _ = nn_influence_utils.compute_influences(
                n_gpu=1,
                device=torch.device("cuda"),
                batch_train_data_loader=dataloader,
                instance_train_data_loader=dataloader,
                model=model.to(device),
                test_inputs=test,
                params_filter=params_filter,
                weight_decay=constants.WEIGHT_DECAY,
                weight_decay_ignores=weight_decay_ignores,
                s_test_damp=s_test_damp,
                s_test_scale=s_test_scale,
                s_test_num_samples=s_test_num_samples,
                train_indices_to_include=KNN_indices)
      
      helpful,harmful = misc_utils.get_helpful_harmful_indices_from_influences_dict(influences)

      # Top k most influential 
      topk_helpful = helpful[:num_influential_sample]
      topk_harmful = harmful[:num_influential_sample]

      for helpful_idx in topk_helpful:
        influence_scores.append(influences[helpful_idx])
        inf_sample = {}
        inf_sample['attention_mask']= torch.tensor(train_instance_dataset_mnli[helpful_idx].attention_mask).to(device)
        inf_sample['input_ids']= torch.tensor(train_instance_dataset_mnli[helpful_idx].input_ids).to(device)
        inf_sample['labels']= torch.tensor(train_instance_dataset_mnli[helpful_idx].label).to(device)
        inf_sample['token_type_ids']= torch.tensor(train_instance_dataset_mnli[helpful_idx].token_type_ids).to(device)
        influence_samples.append(inf_sample)
  return test_samples, influence_samples, influence_scores


