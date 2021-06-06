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


# Metric for evaluation 
def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(task_name, preds, p.label_ids)

    return compute_metrics_fn

def train_augmented(model, augmented_dataset, model_path, training_args = training_args):
  '''
  Fine tune model with augmented dataset

  Args:
    model: trained model to fine tune
    augmented_dataset: augmented dataset to fine tune on
    training_args: training arguments
  '''


  model.load_state_dict(torch.load(model_path))
  # Initialize our Trainer
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=augmented_dataset,
      eval_dataset=eval_instance_dataset_mnli,
      compute_metrics=build_compute_metrics_fn('mnli-2'),
  )
  trainer.train()
  trainer.save_model()

# evaluate
def evaluate(model_path,eval_dataset_name):

  model_mnli.load_state_dict(torch.load(model_path))
  model_mnli.eval()
  output_mode = 'classification'
  # Initialize our Trainer
  trainer = Trainer(
      model=model_mnli,
      args=training_args,
      train_dataset=train_instance_dataset_mnli,
      eval_dataset=eval_instance_dataset_mnli,
      compute_metrics=build_compute_metrics_fn('mnli-2'),
  )

  if eval_dataset_name == 'mnli':
    eval_result = trainer.evaluate(eval_dataset=eval_instance_dataset_mnli)
    print(("***** MNLI Eval results *****"))
  elif eval_dataset_name == 'hans':
    eval_result = trainer.evaluate(eval_dataset=eval_instance_dataset_hans)
    print(("***** HANS Eval results *****"))

  for key, value in eval_result.items():
    print(("%s = %s\n" % (key, value)))
