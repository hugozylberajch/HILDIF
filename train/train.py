import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List
import pandas as pd

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataProcessor,
    # glue_compute_metrics,
    # glue_output_modes,
    # glue_tasks_num_labels,
    set_seed,
)

from transformers import (
    GlueDataset,
    GlueDataTrainingArguments,
)

# FASTIF library
from experiments.data_utils import (
    CustomGlueDataset,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    HansProcessor)

from utils.Processors import *
from utils.Utils import *
from utils.Bert import *

task = 'mnli-2'

if task == "SA":
  sst_processor = Sst2Processor()
  label_list = sst_processor.get_labels()
  num_labels = len(label_list)

model_name_or_path = 'bert-base-cased'
task_name = "mnli-2"
cache_dir = r'./models/mnli/cache'
data_dir = r'./data/mnli'
data_dir_hans = r'./data/hans'
output_dir = r'./models/mnli/mnli_max'
log_dir = r'./log'
do_train = True
do_eval = True
do_predict = False
num_train_epochs = 2
per_device_train_batch_size  = 32
learning_rate  = 2e-5
weight_decay = 0.005
save_steps = 10000
max_seq_length=128


def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(task_name, preds, p.label_ids)

    return compute_metrics_fn

if __name__ == "__main__":

  data_args = GlueDataTrainingArguments(task_name=task_name, 
                                        data_dir=data_dir,
                                        max_seq_length = max_seq_length
  )

  data_args_hans = GlueDataTrainingArguments(task_name='hans', 
                                        data_dir=data_dir_hans,
                                        max_seq_length = max_seq_length
  )


  if not task == 'SA':
    output_mode = glue_output_modes[data_args.task_name]
    num_labels = glue_tasks_num_labels[data_args.task_name]

  training_args = TrainingArguments(output_dir,
                                    do_train=do_train,do_eval=do_eval,
                                    do_predict=do_predict,
                                    evaluation_strategy = 'steps',
                                    num_train_epochs=num_train_epochs, 
                                    per_device_train_batch_size =per_device_train_batch_size, 
                                    learning_rate=learning_rate,
                                    weight_decay = weight_decay,
                                    save_steps = save_steps,
                                    disable_tqdm = False,
                                    eval_steps = 2000)

  config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    finetuning_task= task_name,
    cache_dir= cache_dir,
  )
  tokenizer = AutoTokenizer.from_pretrained(
      model_name_or_path,
      cache_dir=cache_dir,
  )
  model = AutoModelForSequenceClassification.from_pretrained(
      model_name_or_path,
      from_tf=bool(".ckpt" in model_name_or_path),
      config=config,
      cache_dir=cache_dir,
  )

  # Get datasets
  if task == 'SA':
  train_dataset = sst_processor.get_train_examples(data_dir, -1)
  eval_dataset= sst_processor.get_dev_examples(data_dir, -1)

  else:

    train_dataset = (
        CustomGlueDataset(args = data_args, tokenizer=tokenizer, cache_dir=cache_dir)
        if do_train
        else None
    )
    eval_dataset = (
        CustomGlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=cache_dir)
        if do_eval
        else None
    )
    test_dataset = (
        CustomGlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=cache_dir)
        if training_args.do_predict
        else None
    )

  train_dataset_hans = CustomGlueDataset(args = data_args_hans,
                                         tokenizer=tokenizer)

  eval_dataset_hans = CustomGlueDataset(data_args_hans,
                                        tokenizer=tokenizer,
                                        mode="dev")

  # Initialize our Trainer
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      compute_metrics=build_compute_metrics_fn(data_args.task_name),
  )

  # Training
  if training_args.do_train:
      trainer.train()
      trainer.save_model()
