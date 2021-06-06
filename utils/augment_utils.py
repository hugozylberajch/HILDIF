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

def debug_user(test_samples, influence_samples, influence_scores,num_influential_sample = 5):

  '''
  User interface for collecting feedback with given anchor points and influential samples. 

  Args: 
    test_samples: anchor points
    influence_samples: influential samples
    influence_scores: influence scores for the influential samples
    num_influential_sample: number of influential samples per anchor points

  Return: 
    user_scores: user score for each influential sample
  '''

  counter = 0
  user_scores = []
  for test in test_samples:
    sample = {}
    sample['attention_mask']= test['attention_mask'].to(device)
    sample['input_ids']=test['input_ids'].to(device)
    sample['labels']=test['labels'].to(device)
    sample['token_type_ids']=test['token_type_ids'].to(device)
    print('TEST CASE ' + str(counter))
    print('Proposition: ',' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[0])
    print('Hypothesis: ',' '.join(tokenizer.convert_ids_to_tokens(test['input_ids'][0])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[1] )
    print('Class: ', colored('entailment','green') *(test['labels'].item()==1) + colored('non_entailment','red')*(test['labels'].item()==0))
    print('Predicted Class: ',colored('entailment','green') *(model_mnli(**sample).logits[0].argmax().item()==1) + colored('non_entailment','red')*(model_mnli(**sample).logits[0].argmax().item()==0))
    print(' ')

    for i in range(num_influential_sample):
      print('Influential Training Sample ' + str(i))
      print('Influence score: ', influence_scores[i+counter])
      print('Proposition: ',' '.join(tokenizer.convert_ids_to_tokens(influence_samples[i+counter]['input_ids'])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[0])
      print('Hypothesis: ',' '.join(tokenizer.convert_ids_to_tokens(influence_samples[i+counter]['input_ids'])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[1] )
      print('Class: ', colored('entailment','green') *(influence_samples[i+counter]['labels'].item()==1) + colored('non-entailment','red')*(influence_samples[i+counter]['labels'].item()==0))
      print(' ')
    
      text = "The main sample and the presented sample are: (1) Not similar at all (2) Not similar (3) I don't know (4) Quite similar (5) Very similar (1/2/3/4/5)"
      decision = input(text)
      if decision == 1 or decision == 2:
        user_score = 0
      if decision == 3:
        user_score = 0.1
      if decision == 4: 
        user_score = 0.5
      if decision == 5:
        user_score = 0.9
      user_scores.append(user_score)

    counter += num_influential_sample
    
  return user_scores

def augment_with_user(tokenizer, inputs_to_augment, scores):
    '''
    Return augmented dataset with synonym replacement with user feedback. 

    Args: 
      tokenizer: tokenizer of the trained model
      inputs_to_augment: samples to be augmented
      scores: scores from user feedback

    Return: 
      Augmented samples
    '''

    augmented_influential = []
    # DATA AUGMENT
    t_translate = Translate(src="en", to="fr")
    t_synonym = EDA()

    for idx,input in enumerate(inputs_to_augment):
      premise = ' '.join(tokenizer.convert_ids_to_tokens(input.input_ids)).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[0]
      hypothesis = ' '.join(tokenizer.convert_ids_to_tokens(input.input_ids)).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[1]
      label = 'non_entailment'*(input.label==0) + 'entailment'*(input.label==1)
      main_example = InputExample(guid=None, text_a=premise, text_b=hypothesis, label=label)
      augmented_influential.append(main_example) #Add to augmented dataset
      
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

    # for idx,sample in enumerate(corrected_sentences):
    #   premise = sample[0]
    #   hypothesis = sample[1]
    #   label = sample[2]
    #   main_example = InputExample(guid=None, text_a=premise, text_b=hypothesis, label=label)
    #   augmented_influential.append(main_example) #Add to augmented dataset
      
    #   # Backtranslation
    #   # new_translated_premise = t_translate.augment(premise)
    #   # new_translated_hypothesis = t_translate.augment(hypothesis)
    #   # translated_example = InputExample(guid=None, text_a=new_translated_premise, text_b=new_translated_hypothesis, label=label)
    #   # augmented_influential.append(translated_example) #Add to augmented dataset

    #   # Synonym Replacement
    #   for i in range(int(float(scores[idx])*10)):
    #     new_sentence_synonym_premise = t_synonym.synonym_replacement(premise)
    #     new_sentence_synonym_hypothesis = t_synonym.synonym_replacement(hypothesis)
    #     synonym_example = InputExample(guid=None, text_a=new_sentence_synonym_premise, text_b=new_sentence_synonym_hypothesis, label=label)
    #     augmented_influential.append(synonym_example) #Add to augmented dataset

    return augmented_influential

def augment_without_user(tokenizer, inputs_to_augment, num_synonym = 50, test_case = None ):
    '''
    Return augmented dataset with synonym replacement without user feedback. 

    Args: 
      tokenizer: tokenizer of the trained model
      inputs_to_augment: samples to be augmented
      num_synonym: number of generated augmentations
      test_case: name of the testing instance

    Return: 
      Augmented samples
    '''


    if test_case is not None:
      log = open(r'./ISO/log/' +test_case + '.txt', "w")

    augmented_influential = []
    # DATA AUGMENT
    t_translate = Translate(src="en", to="fr")
    t_synonym = EDA()
    for idx,input in enumerate(inputs_to_augment):
      premise = ' '.join(tokenizer.convert_ids_to_tokens(input['input_ids'])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[0]
      hypothesis = ' '.join(tokenizer.convert_ids_to_tokens(input['input_ids'])).replace(' ##','').replace('[CLS]', '').strip().split('[SEP]')[1]
      label = 'non_entailment'*(input['labels']==0) + 'entailment'*(input['labels']==1)
      main_example = InputExample(guid=None, text_a=premise, text_b=hypothesis, label=label)
      augmented_influential.append(main_example) #Add to augmented dataset
      log.write('INFLUENTIAL SAMPLE ' + str(idx) +'\n')
      log.write('Proposition: '+premise+'\n')
      log.write('Hypothesis: '+hypothesis+'\n')
      log.write('Class: '+ label+'\n')
      
      # Backtranslation
      # new_translated_premise = t_translate.augment(premise)
      # new_translated_hypothesis = t_translate.augment(hypothesis)
      # translated_example = InputExample(guid=None, text_a=new_translated_premise, text_b=new_translated_hypothesis, label=label)
      # augmented_influential.append(translated_example) #Add to augmented dataset

      # Synonym Replacement
      for i in range(num_synonym):
        new_sentence_synonym_premise = t_synonym.synonym_replacement(premise)
        new_sentence_synonym_hypothesis = t_synonym.synonym_replacement(hypothesis)
        synonym_example = InputExample(guid=None, text_a=new_sentence_synonym_premise, text_b=new_sentence_synonym_hypothesis, label=label)
        augmented_influential.append(synonym_example) #Add to augmented dataset

        # Backtranslation
        # new_synonym_translated_premise = t_translate.augment(new_sentence_synonym_premise)
        # new_synonym_translated_hypothesis = t_translate.augment(new_sentence_synonym_hypothesis)
        # synonym_translated_example = InputExample(guid=None, text_a=new_synonym_translated_premise, text_b=new_synonym_translated_hypothesis, label=label)
        # augmented_influential.append(synonym_translated_example) #Add to augmented dataset

    random.shuffle(augmented_influential)
    return glue_convert_examples_to_features(
                  augmented_influential,
                  tokenizer,
                  max_length=max_seq_length,
                  label_list=["non_entailment", "entailment"],
                  output_mode="classification",
              )



