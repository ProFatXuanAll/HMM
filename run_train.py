
# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re

# 3rd-party modules

import numpy as np

# self made modules

import first_order_hmm

dataset = first_order_hmm.dataset.NERDataset()

(
    token_map,
    inv_token_map,
    label_map,
    inv_label_map
) = first_order_hmm.preprocess.build_token_and_label_lookup(dataset=dataset)

(
    emit_prob_mat,
    init_prob_mat,
    tran_prob_mat
) = first_order_hmm.preprocess.build_prob_mat(
    dataset=dataset,
    token_map=token_map,
    label_map=label_map
)

model = first_order_hmm.model.FirstOrderHMM(
    emit_prob_mat=emit_prob_mat,
    init_prob_mat=init_prob_mat,
    num_observations=emit_prob_mat.shape[1],
    num_states=init_prob_mat.shape[0],
    tran_prob_mat=tran_prob_mat
)

# Training loop.
for _ in range(1):
    for sequence, label in dataset:
        tokens = re.split(r'\s+', sequence)
        obs_indices = np.array([token_map[token] for token in tokens])
        model.baum_welch(obs_indices=obs_indices)

# Inference loop.
for sequence, label in dataset:
    tokens = re.split(r'\s+', sequence)
    obs_indices = np.array([token_map[token] for token in tokens])

    _, pred_label_idx = model.viterbi(obs_indices=obs_indices)
    pred_label = [inv_label_map[label_idx] for label_idx in pred_label_idx]
    print('===============================')
    print(sequence)
    print(label)
    print(pred_label)
    print('===============================')
