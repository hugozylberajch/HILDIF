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
        
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              guid=example.guid))
    return features

# adapted from AllenNLP Interpret
def _register_embedding_list_hook(model, embeddings_list, model_type):
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())
    if model_type == 'BERT':
        embedding_layer = model.bert.embeddings.word_embeddings
    elif model_type == 'LSTM':
        embedding_layer = model.my_word_embeddings
    else:
        raise ValueError("Current model type not supported.")
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle

def _register_embedding_gradient_hooks(model, embeddings_gradients, model_type):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0])
    if model_type == 'BERT':
        embedding_layer = model.bert.embeddings.word_embeddings
    elif model_type == 'LSTM':
        embedding_layer = model.my_word_embeddings
    else:
        raise ValueError("Current model type not supported.")
    hook = embedding_layer.register_backward_hook(hook_layers)
    return hook

def saliency_map(model, input_ids, segment_ids, input_mask, pred_label_ids, model_type='BERT'):
    embeddings_list = []
    handle = _register_embedding_list_hook(model, embeddings_list, model_type)
    embeddings_gradients = []
    hook = _register_embedding_gradient_hooks(model, embeddings_gradients, model_type)

    model.zero_grad()
    _loss = model(input_ids, segment_ids, input_mask, pred_label_ids)
    _loss.backward()
    handle.remove()
    hook.remove()

    saliency_grad = embeddings_gradients[0].detach().cpu().numpy()
    saliency_grad = np.sum(saliency_grad[0] * embeddings_list[0], axis=1)
    norm = np.linalg.norm(saliency_grad, ord=1)
#     saliency_grad = [math.fabs(e) / norm for e in saliency_grad]
    saliency_grad = [(- e) / norm for e in saliency_grad] # negative gradient for loss means positive influence on decision
    return saliency_grad

################ functions for influence function ################

def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

def unflatten_to_param_dim(x, param_shape_tensor):
    tar_p = []
    ptr = 0
    for p in param_shape_tensor:
        len_p = torch.numel(p)
        tmp = x[ptr : ptr + len_p].view(p.shape)
        tar_p.append(tmp)
        ptr += len_p
    return tar_p

def hv(loss, model_params, v): # according to pytorch issue #24004
#     s = time.time()
    grad = autograd.grad(loss, model_params, create_graph=True, retain_graph=True)
#     e1 = time.time()
    Hv = autograd.grad(grad, model_params, grad_outputs=v)
#     e2 = time.time()
#     print('1st back prop: {} sec. 2nd back prop: {} sec'.format(e1-s, e2-e1))
    return Hv

######## LiSSA ########

def get_inverse_hvp_lissa(v, model, device, param_influence, train_loader, damping, num_samples, recursion_depth, scale=1e4):
    ihvp = None
    for i in range(num_samples):
        cur_estimate = v
        lissa_data_iterator = iter(train_loader)
        for j in tqdm(range(recursion_depth)):
            try:
                input_ids, input_mask, segment_ids, label_ids, guids = next(lissa_data_iterator)
            except StopIteration:
                lissa_data_iterator = iter(train_loader)
                input_ids, input_mask, segment_ids, label_ids, guids = next(lissa_data_iterator)
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            model.zero_grad()
            train_loss = model(input_ids, segment_ids, input_mask, label_ids)
            hvp = hv(train_loss, param_influence, cur_estimate)
            cur_estimate = [_a + (1 - damping) * _b - _c / scale for _a, _b, _c in zip(v, cur_estimate, hvp)]
            if (j % 200 == 0) or (j == recursion_depth - 1):
                print("Recursion at depth %s: norm is %f" % (j, np.linalg.norm(gather_flat_grad(cur_estimate).cpu().numpy())))
        if ihvp == None:
            ihvp = [_a / scale for _a in cur_estimate]
        else:
            ihvp = [_a + _b / scale for _a, _b in zip(ihvp, cur_estimate)]
    return_ihvp = gather_flat_grad(ihvp)
    return_ihvp /= num_samples
    return return_ihvp

def accuracy(out, label_ids):
    # axis-0: seqs in batch; axis-1: potential labels of seq
    outputs = np.argmax(out, axis=1)
    matched = outputs == label_ids
    num_correct = np.sum(matched)
    num_total = len(label_ids)
    return num_correct, num_total
    
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
