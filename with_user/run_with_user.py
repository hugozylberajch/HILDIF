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

#FASTIF
from experiments import mnli
from experiments import hans
from experiments import s_test_speedup
from experiments import misc_utils
from experiments import constants
from experiments import influence_helpers
from influence_utils import faiss_utils
from influence_utils import nn_influence_utils
from experiments.data_utils import (
    CustomGlueDataset,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    HansProcessor)

from Utils import saliency_map

from influence_utils import get_test, get_influence
from augment_utils import debug_user, augment_with_user
from train_utils import train_augmented
from log_utils import get_test_from_log, get_inf_from_log, get_influence_from_log, debug_from_log, augment_from_log_with_user

# CONSTANTS

# Paths
HANS_MODEL_PATH = r'./models/hans/'
MNLI_MODEL_PATH = r'./models/mnli/'
data_dir_mnli = r'./data/mnli'
data_dir_hans = r'./data/hans'
output_dir = r'./models/mnli/mnli_augmented'
log_dir = r'./log/mnli/mnli_augmented'

# Data
max_seq_length = 128
data_args_mnli = GlueDataTrainingArguments(task_name='mnli-2', 
                                      data_dir=data_dir_mnli,
                                      max_seq_length = max_seq_length
)

data_args_hans = GlueDataTrainingArguments(task_name='hans', 
                                      data_dir=data_dir_hans,
                                      max_seq_length = max_seq_length
)

# Training
do_train = True
do_eval = True
do_predict = False
num_train_epochs = 2
per_device_train_batch_size = 16
learning_rate = 1e-5
weight_decay = 0.005
save_steps = 50
eval_steps = 50

training_args = TrainingArguments(output_dir,
                                  do_train=do_train,
                                  do_eval=do_eval,
                                  do_predict=do_predict,
                                  evaluation_strategy = 'steps',
                                  num_train_epochs=num_train_epochs, 
                                  per_device_train_batch_size =per_device_train_batch_size, 
                                  learning_rate=learning_rate,
                                  weight_decay = weight_decay,
                                  save_steps = save_steps,
                                  disable_tqdm = False,
                                  eval_steps = eval_steps)

# Influence function
k = 10 #Top k
num_synonym = 100

# Setting random seed and device
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#Influence calculation
s_test_damp, s_test_scale, s_test_num_samples = influence_helpers.select_s_test_config(
        trained_on_task_name="mnli-2",
        train_task_name="mnli-2",
        eval_task_name="mnli-2")

def from_log_retrain_with_user(log, num_test_case, init_num_influential_sample, num_influential_sample,save_steps):

      output_dir = r'./models/mnli/mnli_augmented/' + log + '_user'
      # Create output directory
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)

      training_args = TrainingArguments(output_dir,
                                      do_train=do_train,
                                      do_eval=do_eval,
                                      do_predict=do_predict,
                                      evaluation_strategy = 'steps',
                                      num_train_epochs=num_train_epochs, 
                                      per_device_train_batch_size =per_device_train_batch_size, 
                                      learning_rate=learning_rate,
                                      weight_decay = weight_decay,
                                      save_steps = save_steps,
                                      disable_tqdm = False,
                                      eval_steps = eval_steps)
      
      augmented_dataset = augment_from_log_with_user(log, num_test_case, init_num_influential_sample, num_influential_sample=num_influential_sample)
      train_augmented(augmented_dataset, model_path = r'./models/mnli/pytorch_model.bin', training_args = training_args)

if __name__ == "__main__":
    config_mnli = AutoConfig.from_pretrained(
        MNLI_MODEL_PATH,
        num_labels=2,
        finetuning_task= 'mnli-2'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-cased'
    )
    model_mnli = AutoModelForSequenceClassification.from_pretrained(
        MNLI_MODEL_PATH,
        from_tf=bool(".ckpt" in MNLI_MODEL_PATH),
        config=config_mnli
    )

    train_instance_dataset_hans = CustomGlueDataset(args = data_args_hans,
                                                    tokenizer=tokenizer)
    eval_instance_dataset_hans = CustomGlueDataset(data_args_hans,
                                                   tokenizer=tokenizer,
                                                   mode="dev")
    train_instance_dataset_mnli = CustomGlueDataset(data_args_mnli,
                                                    tokenizer=tokenizer)
    eval_instance_dataset_mnli = CustomGlueDataset(data_args_mnli,
                                                   tokenizer=tokenizer,
                                                   mode="dev")

    train_instance_data_loader_mnli = misc_utils.get_dataloader(
            train_instance_dataset_mnli,
            batch_size=1,
            random=True)
    eval_instance_data_loader_mnli = misc_utils.get_dataloader(
            eval_instance_dataset_mnli,
            batch_size=1,
            random=True)
    train_instance_data_loader_hans = misc_utils.get_dataloader(
            train_instance_dataset_hans,
            batch_size=1)
    eval_instance_data_loader_hans = misc_utils.get_dataloader(
            eval_instance_dataset_hans,
            batch_size=1)

    # Import Faiss Indices
    faiss_index = faiss_utils.FAISSIndex(768, "Flat")
    faiss_index_mnli = faiss_utils.FAISSIndex(768, "Flat")
    faiss_index_mnli.load(MNLI_MODEL_PATH + 'index.faiss')
    # faiss_index.load(HANS_MODEL_PATH + 'index.faiss')

    #Model settings
    params_filter = [
        n for n, p in model_mnli.named_parameters()
        if not p.requires_grad]
    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in model_mnli.named_parameters()
        if not p.requires_grad]

    # Selecting a log to augment from
    log = 'Target_test_20_inf_7_test_50_syn'

    # Run experiment without user
    from_log_retrain_with_user(log, num_test_case = 20, init_num_influential_sample = 7, num_influential_sample = 7, save_steps = 500)
