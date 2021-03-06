
# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Tuple

# 3rd-party modules

import numpy as np


class FirstOrderHMM:
    def __init__(
            self,
            emit_prob_mat: np.ndarray,
            init_prob_mat: np.ndarray,
            num_states: int,
            num_observations: int,
            tran_prob_mat: np.ndarray
    ):
        # Type check.
        if not isinstance(emit_prob_mat, np.ndarray):
            raise TypeError(
                '`emit_prob_mat` must be an instance of `np.ndarray`.'
            )

        if not isinstance(init_prob_mat, np.ndarray):
            raise TypeError(
                '`init_prob_mat` must be an instance of `np.ndarray`.'
            )

        if not isinstance(num_states, int):
            raise TypeError('`num_states` must be an instance of `int`.')

        if not isinstance(num_observations, int):
            raise TypeError('`num_observations` must be an instance of `int`.')

        if not isinstance(tran_prob_mat, np.ndarray):
            raise TypeError(
                '`tran_prob_mat` must be an instance of `np.ndarray`.'
            )

        # Value check.
        assert num_states >= 1
        assert num_observations >= 1
        assert init_prob_mat.shape == (num_states,)
        assert emit_prob_mat.shape == (num_states, num_observations)
        assert tran_prob_mat.shape == (num_states, num_states)
        assert np.all((init_prob_mat >= 0) & (init_prob_mat <= 1))
        assert np.all((emit_prob_mat >= 0) & (emit_prob_mat <= 1))
        assert np.all((tran_prob_mat >= 0) & (tran_prob_mat <= 1))

        self.emit_prob_mat = emit_prob_mat
        self.init_prob_mat = init_prob_mat
        self.num_states = num_states
        self.num_observations = num_observations
        self.tran_prob_mat = tran_prob_mat

    def forward(self, obs_indices: np.ndarray) -> np.ndarray:
        total_time_step = obs_indices.shape[0]
        forward_prob_mat = np.zeros((total_time_step, self.num_states))

        # Initialize time step 0.
        forward_prob_mat[0] = self.init_prob_mat * \
            self.emit_prob_mat[:, obs_indices[0]]

        # Iteratively calculate forward probabilities \alpha_t(i).
        for time_step in range(1, total_time_step):
            forward_prob_vec = (
                forward_prob_mat[time_step - 1].reshape(-1, 1) *
                self.tran_prob_mat
            ).sum(axis=0)
            forward_prob_mat[time_step] = (
                forward_prob_vec *
                self.emit_prob_mat[:, obs_indices[time_step]]
            )

        return forward_prob_mat

    def obs_prob_by_forward(self, obs_indices: np.ndarray) -> float:
        forward_prob_mat = self.forward(obs_indices=obs_indices)
        return forward_prob_mat[-1].sum()

    def backward(self, obs_indices: np.ndarray) -> np.ndarray:
        total_time_step = obs_indices.shape[0]
        backward_prob_mat = np.zeros((total_time_step, self.num_states))

        # Initialize last time step T.
        backward_prob_mat[-1] = 1

        # Iteratively calculate backward probabilities \beta_t(i).
        for time_step in range(total_time_step - 2, -1, -1):
            backward_prob_mat[time_step] = (
                self.tran_prob_mat *
                self.emit_prob_mat[:, obs_indices[time_step + 1]].reshape(1, -1) *
                backward_prob_mat[time_step + 1].reshape(1, -1)
            ).sum(axis=1)

        return backward_prob_mat

    def obs_prob_by_backward(self, obs_indices: np.ndarray) -> float:
        backward_prob_mat = self.backward(obs_indices=obs_indices)
        return (
            self.init_prob_mat *
            self.emit_prob_mat[:, obs_indices[0]] *
            backward_prob_mat[0]
        ).sum()

    def viterbi(
            self,
            obs_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        total_time_step = obs_indices.shape[0]
        viterbi_prob_mat = np.zeros((total_time_step, self.num_states))
        prev_state_mat = np.zeros(
            (total_time_step, self.num_states), dtype=np.int64)

        # Initialize time step 0.
        viterbi_prob_mat[0] = self.init_prob_mat * \
            self.emit_prob_mat[:, obs_indices[0]]

        # Iteratively calculate Viterbi probabilities \delta_t(i) and previous best state
        # \psi_t(i).
        for time_step in range(1, total_time_step):
            viterbi_prob_vec = (
                viterbi_prob_mat[time_step - 1].reshape(-1, 1) *
                self.tran_prob_mat
            )
            prev_state_mat[time_step] = np.argmax(viterbi_prob_vec, axis=0)
            viterbi_prob_vec = np.max(viterbi_prob_vec, axis=0)
            viterbi_prob_mat[time_step] = (
                viterbi_prob_vec *
                self.emit_prob_mat[:, obs_indices[time_step]]
            )

        # Back track all best states.
        all_best_states = [np.argmax(prev_state_mat[-1])]
        for time_step in range(total_time_step - 2, -1, -1):
            all_best_states.append(
                prev_state_mat[time_step, all_best_states[-1]])

        all_best_states.reverse()

        return viterbi_prob_mat, np.array(all_best_states)

    def baum_welch(self, obs_indices: np.ndarray) -> None:
        total_time_step = obs_indices.shape[0]
        forward_prob_mat = self.forward(obs_indices=obs_indices)
        backward_prob_mat = self.backward(obs_indices=obs_indices)
        obs_prob = self.obs_prob_by_forward(obs_indices=obs_indices)

        # Iteratively calculate expected number of times transition from state i to state j at time
        # step t \xi_t(i, j). Also calculate expected number of times transistion from state i at
        # time step t \gamma_t(i).
        exp_from_i_to_j = np.zeros(
            (total_time_step - 1, self.num_states, self.num_states))
        exp_from_i = np.zeros((total_time_step, self.num_states))

        for time_step in range(total_time_step - 1):
            exp_from_i_to_j[time_step] = (
                forward_prob_mat[time_step].reshape(-1, 1) *
                self.tran_prob_mat *
                self.emit_prob_mat[:, obs_indices[time_step + 1]].reshape(1, -1) *
                backward_prob_mat[time_step + 1].reshape(1, -1)
            ) / obs_prob
            exp_from_i[time_step] = exp_from_i_to_j[time_step].sum(axis=-1)

        exp_from_i[-1] = forward_prob_mat[-1] * \
            backward_prob_mat[-1] / obs_prob

        # Update parameters.
        self.init_prob_mat = exp_from_i[0]
        self.tran_prob_mat = (
            exp_from_i_to_j.sum(axis=0) /
            exp_from_i[:-1].sum(axis=0).reshape(-1, 1)
        )
        self.emit_prob_mat = (
            exp_from_i.T @ (
                obs_indices.reshape(-1, 1) ==
                np.arange(self.num_observations).reshape(1, -1)
            ) /
            exp_from_i.sum(axis=0).reshape(-1, 1)
        )


class FirstOrderHMMDumb(FirstOrderHMM):
    def forward(self, obs_indices: np.ndarray) -> np.ndarray:
        total_time_step = obs_indices.shape[0]
        forward_prob_mat = np.zeros((total_time_step, self.num_states))

        # Initialize time step 0.
        forward_prob_mat[0] = self.init_prob_mat * \
            self.emit_prob_mat[:, obs_indices[0]]

        # Iteratively calculate forward probabilities \alpha_t(i).
        for time_step in range(1, total_time_step):
            for j in range(self.num_states):
                forward_prob = 0.0

                for i in range(self.num_states):
                    forward_prob += (
                        forward_prob_mat[time_step - 1, i] *
                        self.tran_prob_mat[i, j]
                    )

                forward_prob_mat[time_step, j] = (
                    forward_prob *
                    self.emit_prob_mat[j, obs_indices[time_step]]
                )

        return forward_prob_mat

    def backward(self, obs_indices: np.ndarray) -> np.ndarray:
        total_time_step = obs_indices.shape[0]
        backward_prob_mat = np.zeros((total_time_step, self.num_states))

        # Initialize last time step T.
        backward_prob_mat[-1] = 1

        # Iteratively calculate backward probabilities \beta_t(i).
        for time_step in range(total_time_step - 2, -1, -1):
            for i in range(self.num_states):
                backward_prob = 0.0
                for j in range(self.num_states):
                    backward_prob += (
                        self.tran_prob_mat[i, j] *
                        self.emit_prob_mat[j, obs_indices[time_step + 1]] *
                        backward_prob_mat[time_step + 1, j]
                    )
                backward_prob_mat[time_step, i] = backward_prob

        return backward_prob_mat

    def viterbi(
            self,
            obs_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        total_time_step = obs_indices.shape[0]
        viterbi_prob_mat = np.zeros((total_time_step, self.num_states))
        prev_state_mat = np.zeros(
            (total_time_step, self.num_states), dtype=np.int64)

        # Initialize time step 0.
        viterbi_prob_mat[0] = self.init_prob_mat * \
            self.emit_prob_mat[:, obs_indices[0]]

        # Iteratively calculate Viterbi probabilities \delta_t(i) and previous best state
        # \psi_t(i).
        for time_step in range(1, total_time_step):
            for j in range(self.num_states):
                viterbi_prob = 0.0
                prev_state = 0

                for i in range(self.num_states):
                    candidate_viterbi_prob = (
                        viterbi_prob_mat[time_step - 1, i] *
                        self.tran_prob_mat[i, j]
                    )

                    if viterbi_prob < candidate_viterbi_prob:
                        viterbi_prob = candidate_viterbi_prob
                        prev_state = i

                viterbi_prob_mat[time_step, j] = (
                    viterbi_prob *
                    self.emit_prob_mat[j, obs_indices[time_step]]
                )
                prev_state_mat[time_step, j] = prev_state

        # Back track all best states.
        all_best_states = [np.argmax(prev_state_mat[-1])]
        for time_step in range(total_time_step - 2, -1, -1):
            all_best_states.append(
                prev_state_mat[time_step, all_best_states[-1]])

        all_best_states.reverse()

        return viterbi_prob_mat, np.array(all_best_states)

    def baum_welch(self, obs_indices: np.ndarray) -> None:
        total_time_step = obs_indices.shape[0]
        forward_prob_mat = self.forward(obs_indices=obs_indices)
        backward_prob_mat = self.backward(obs_indices=obs_indices)
        obs_prob = self.obs_prob_by_forward(obs_indices=obs_indices)

        # Iteratively calculate expected number of times transition from state i to state j at time
        # step t \xi_t(i, j). Also calculate expected number of times transistion from state i at
        # time step t \gamma_t(i).
        exp_from_i_to_j = np.zeros(
            (total_time_step - 1, self.num_states, self.num_states))
        exp_from_i = np.zeros((total_time_step, self.num_states))

        for time_step in range(total_time_step - 1):
            for i in range(self.num_states):
                for j in range(self.num_states):
                    exp_from_i_to_j[time_step, i, j] = (
                        forward_prob_mat[time_step, i] *
                        self.tran_prob_mat[i, j] *
                        self.emit_prob_mat[j, obs_indices[time_step + 1]] *
                        backward_prob_mat[time_step + 1, j]
                    ) / obs_prob
                exp_from_i[time_step, i] = exp_from_i_to_j[time_step, i].sum()

        exp_from_i[-1] = forward_prob_mat[-1] * \
            backward_prob_mat[-1] / obs_prob

        # Update parameters.
        for i in range(self.num_states):
            self.init_prob_mat[i] = exp_from_i[0, i]
            for j in range(self.num_states):
                self.tran_prob_mat[i, j] = (
                    exp_from_i_to_j[:, i, j].sum() /
                    exp_from_i[:-1, i].sum()
                )
            for k in range(self.num_observations):
                self.emit_prob_mat[i, k] = (
                    ((obs_indices == k) * exp_from_i[:, i]).sum() /
                    exp_from_i[:, i].sum()
                )
