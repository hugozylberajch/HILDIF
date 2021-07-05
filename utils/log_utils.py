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

def get_test_from_log(log, num_test_case):
  test_examples = []
  log = open(r'/content/drive/My Drive/ISO/log/'+log+'_samples.txt', "r")
  a,b,c = False,False,False
  counter = 0
  for i,line in enumerate(log):
    if line[:3] == 'Pro':
      proposition = line[13:]
      a = True
    if line[:3] == 'Hyp':
      hypothesis = line[13:]
      b = True
    if line[:3] == 'Cla':
      label = line[8:-1]
      c = True
    if a and b and c:
      test = InputExample(guid=None, text_a=proposition, text_b=hypothesis, label=label)
      test_examples.append(test)
      a,b,c = False,False,False
      counter += 1 
    if counter == num_test_case : 
      break
  return test_examples


# To repo
def get_inf_from_log(log):
  inf_examples = []
  log = open(r'/content/drive/My Drive/ISO/log/'+log+'_inf_samples.txt', "r")
  a,b,c = False,False,False
  for i,line in enumerate(log):
    if line[:3] == 'Pro':
      proposition = line[13:]
      a = True
    if line[:3] == 'Hyp':
      hypothesis = line[13:]
      b = True
    if line[:3] == 'Cla':
      label = line[7:-1]
      c = True
    if a and b and c:
      inf = InputExample(guid=None, text_a=proposition, text_b=hypothesis, label=label)
      inf_examples.append(inf)
      a,b,c = False,False,False
  return inf_examples

def get_influence_from_log(log,num_influential_sample = 20, k=3000):

  log2 = open(r'/content/drive/My Drive/ISO/log/' +log + '_inf_samples.txt', "w")

  test_samples = glue_convert_examples_to_features(
                  get_test_from_log(log, num_test_case=10),
                  tokenizer,
                  max_length=max_seq_length,
                  label_list=["non_entailment", "entailment"],
                  output_mode="classification",
              )
  model_mnli.eval()
  counter = 0
  influence_scores = []
  influence_samples = []

  # FIND KNN
  for test in test_samples:
      print(test)
      sample = {}
      sample['attention_mask']= torch.tensor([test.attention_mask]).to(device)
      sample['input_ids']=torch.tensor([test.input_ids]).to(device)
      sample['labels']=torch.tensor([test.label]).to(device)
      sample['token_type_ids']=torch.tensor([test.token_type_ids]).to(device)
      features = misc_utils.compute_BERT_CLS_feature(model_mnli, **sample)
      features = features.cpu().detach().numpy()
      KNN_distances, KNN_indices = faiss_index_mnli.search(
                k=k, queries=features)
      
      # GET INFLUENTIAL TRAINING
      influences, index, _ = nn_influence_utils.compute_influences(
                n_gpu=1,
                device=torch.device("cuda"),
                batch_train_data_loader=train_instance_data_loader_mnli,
                instance_train_data_loader=train_instance_data_loader_mnli,
                model=model_mnli.to(device),
                test_inputs=sample,
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

      for idx,helpful_idx in enumerate(topk_helpful):
        influence_scores.append(influences[helpful_idx])
        inf_sample = {}
        inf_sample['attention_mask']= torch.tensor(train_instance_dataset_mnli[helpful_idx].attention_mask).to(device)
        inf_sample['input_ids']= torch.tensor(train_instance_dataset_mnli[helpful_idx].input_ids).to(device)
        inf_sample['labels']= torch.tensor(train_instance_dataset_mnli[helpful_idx].label).to(device)
        inf_sample['token_type_ids']= torch.tensor(train_instance_dataset_mnli[helpful_idx].token_type_ids).to(device)
        influence_samples.append(inf_sample)
        premise = ' '.join(tokenizer.convert_ids_to_tokens(inf_sample['input_ids'])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[0]
        hypothesis = ' '.join(tokenizer.convert_ids_to_tokens(inf_sample['input_ids'])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[1]
        label = 'non_entailment'*(inf_sample['labels']==0) + 'entailment'*(inf_sample['labels']==1)
        log2.write('INFLUENTIAL SAMPLE ' + str(idx) +'\n')
        log2.write('Proposition: '+premise+'\n')
        log2.write('Hypothesis: '+hypothesis+'\n')
        log2.write('Class: '+ label+'\n')
  return test_samples, influence_samples, influence_scores

def debug_from_log(log, num_test_case, init_num_influential_sample, num_influential_sample):
  counter = 0
  user_scores = []
  corrected_sentences = []
  test_samples = get_test_from_log(log, num_test_case)
  influence_samples = get_inf_from_log(log)
  for idx,test in enumerate(test_samples):
    print('TEST CASE ' + str(idx))
    print('Proposition: ', test.text_a)
    print('Hypothesis: ',test.text_b)
    print(' ')

    for i in range(num_influential_sample):
      print('Influential Training Sample ' + str(i))
      print('Proposition: ',influence_samples[i+counter].text_a)
      print('Hypothesis: ',influence_samples[i+counter].text_b)
      print(' ')
    
      text = "The main sample and the presented sample are: (1) Not similar at all (2) Not similar (3) I don't know (4) Quite similar (5) Very similar (1/2/3/4/5)"
      decision = int(input(text))
      if decision == 1:
        user_score = 10
      if decision == 2:
        user_score = 30
      if decision == 3:
        user_score = 50
      if decision == 4: 
        user_score = 80
      if decision == 5:
        user_score = 100
      user_scores.append(user_score)

      # decision = input('Modify training sample? (Y/N)')
      # if decision == 'Y':
      #   premise = input('New premise: ')
      #   hypothesis = input('New hypothesis: ')
      #   label = input('New Label (entailement/non_entailment): ')
      #   corrected_sentences.append((premise,hypothesis,label))

    counter += init_num_influential_sample
    
  return user_scores, corrected_sentences

def augment_from_log_with_user(log, num_test_case, init_num_influential_sample, num_influential_sample):
    augmented_influential = []
    # DATA AUGMENT
    t_translate = Translate(src="en", to="fr")
    t_synonym = EDA()
    inputs_to_augment = get_inf_from_log(log)
    scores, corrected_sentences = debug_from_log(log, num_test_case, init_num_influential_sample, num_influential_sample=num_influential_sample)
    for idx,input in enumerate(inputs_to_augment):
      premise = input.text_a
      hypothesis = input.text_b
      label = input.label
      augmented_influential.append(input) #Add to augmented dataset
      
      # Backtranslation
      # new_translated_premise = t_translate.augment(premise)
      # new_translated_hypothesis = t_translate.augment(hypothesis)
      # translated_example = InputExample(guid=None, text_a=new_translated_premise, text_b=new_translated_hypothesis, label=label)
      # augmented_influential.append(translated_example) #Add to augmented dataset

      # Synonym Replacement
      for i in range(int(scores[idx])):
        new_sentence_synonym_premise = t_synonym.synonym_replacement(premise)
        new_sentence_synonym_hypothesis = t_synonym.synonym_replacement(hypothesis)
        synonym_example = InputExample(guid=None, text_a=new_sentence_synonym_premise, text_b=new_sentence_synonym_hypothesis, label=label)
        augmented_influential.append(synonym_example) #Add to augmented dataset

        # Backtranslation
        # new_synonym_translated_premise = t_translate.augment(new_sentence_synonym_premise)
        # new_synonym_translated_hypothesis = t_translate.augment(new_sentence_synonym_hypothesis)
        # synonym_translated_example = InputExample(guid=None, text_a=new_synonym_translated_premise, text_b=new_synonym_translated_hypothesis, label=label)
        # augmented_influential.append(synonym_translated_example) #Add to augmented dataset

    for idx,sample in enumerate(corrected_sentences):
      premise = sample[0]
      hypothesis = sample[1]
      label = sample[2]
      main_example = InputExample(guid=None, text_a=premise, text_b=hypothesis, label=label)
      augmented_influential.append(main_example) #Add to augmented dataset
      
      # Backtranslation
      # new_translated_premise = t_translate.augment(premise)
      # new_translated_hypothesis = t_translate.augment(hypothesis)
      # translated_example = InputExample(guid=None, text_a=new_translated_premise, text_b=new_translated_hypothesis, label=label)
      # augmented_influential.append(translated_example) #Add to augmented dataset

      # Synonym Replacement
      for i in range(int(float(scores[idx])*10)):
        new_sentence_synonym_premise = t_synonym.synonym_replacement(premise)
        new_sentence_synonym_hypothesis = t_synonym.synonym_replacement(hypothesis)
        synonym_example = InputExample(guid=None, text_a=new_sentence_synonym_premise, text_b=new_sentence_synonym_hypothesis, label=label)
        augmented_influential.append(synonym_example) #Add to augmented dataset

    return glue_convert_examples_to_features(
                  augmented_influential,
                  tokenizer,
                  max_length=max_seq_length,
                  label_list=["non_entailment", "entailment"],
                  output_mode="classification",
              )
