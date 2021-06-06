# Adapted from the influence-function-analysis repo
# For more details, see https://github.com/xhan77/influence-function-analysis

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import pickle
import time
import math

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

import torch.autograd as autograd
from scipy import stats


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, note=""):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.note = note
        
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid
        
class MnliProcessor(object):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, num_train_samples=-1):
        """See base class."""
        if num_train_samples != -1:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "mnli_train.tsv")), "mnli_train")[: num_train_samples]
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "mnli_train.tsv")), "mnli_train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "mnli_dev.tsv")), "mnli_dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "non-entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            if label == "contradiction" or label == "neutral":
                label = "non-entailment" # collapse contradiction into non-entailment
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class HansProcessor(object):

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "small_heuristics_evaluation_set.txt")), "HANS small")
    
    def get_neg_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "negated_small_heuristics_evaluation_set.txt")), "HANS small negated")

    def get_labels(self):
        """See base class."""
        return ["entailment", "non-entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            text_a = line[5]
            text_b = line[6]
            label = line[0]
            note = line[8]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, note=note))
        return examples

    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class Sst2Processor(object):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir, num_train_samples=-1):
        """See base class."""
        if num_train_samples != -1:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "sst2_train.tsv")), "train")[: num_train_samples]
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "sst2_train.tsv")), "train")

    def get_dev_examples(self, data_dir, num_test_samples =-1):
        """See base class."""
        if num_test_samples != -1:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "sst2_dev.tsv")), "dev")[: num_test_samples]
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "sst2_dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
    
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()