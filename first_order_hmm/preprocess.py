
# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re

from typing import Dict
from typing import Tuple
from typing import Union

# 3rd-party modules

import numpy as np

# self-made modules

from first_order_hmm.dataset import NERDataset, POSDataset

def build_token_and_label_lookup(
        dataset: Union[NERDataset, POSDataset]
) -> Tuple[
    Dict[str, int],
    Dict[int, str],
    Dict[str, int],
    Dict[int, str]
]:
    r"""Return token and label lookup and inverse lookup table."""
    uniq_tokens = set()
    uniq_labels = set()
    for sequence, labels in dataset:
        for token in re.split(r'\s+', sequence):
            uniq_tokens.add(token)
        for label in labels:
            uniq_labels.add(label)

    token_map = {}
    inv_token_map = {}
    label_map = {}
    inv_label_map = {}

    for token_idx, token in enumerate(uniq_tokens):
        token_map[token] = token_idx
        inv_token_map[token_idx] = token
    for label_idx, label in enumerate(uniq_labels):
        label_map[label] = label_idx
        inv_label_map[label_idx] = label

    return token_map, inv_token_map, label_map, inv_label_map


def build_prob_mat(
        dataset: Union[NERDataset, POSDataset],
        token_map: Dict[str, int],
        label_map: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return emition, initial and transition probabilities matrices."""
    emit_prob_mat = np.ones((len(label_map), len(token_map)))
    init_prob_mat = np.ones((len(label_map),))
    tran_prob_mat = np.ones((len(label_map), len(label_map)))

    for sequence, labels in dataset:
        tokens = re.split(r'\s+', sequence)
        init_prob_mat[label_map[labels[0]]] += 1
        for token, label in zip(tokens, labels):
            emit_prob_mat[label_map[label], token_map[token]] += 1

        for label_prev, label_next in zip(labels[:-1], labels[1:]):
            tran_prob_mat[label_map[label_prev], label_map[label_next]] += 1

    emit_prob_mat = emit_prob_mat / emit_prob_mat.sum(axis=-1).reshape(-1, 1)
    init_prob_mat = init_prob_mat / init_prob_mat.sum()
    tran_prob_mat = tran_prob_mat / tran_prob_mat.sum(axis=-1).reshape(-1, 1)

    return emit_prob_mat, init_prob_mat, tran_prob_mat
