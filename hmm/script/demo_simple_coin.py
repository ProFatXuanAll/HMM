import argparse
import sys
from typing import List

from tqdm import tqdm

import hmm.algo.base
import hmm.algo.mat
import hmm.model.coin
import hmm.util.rand


def parse_args(argv: List[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser('python -m hmm.script.demo_simple_coin')
  parser.add_argument(
    '--seq_len',
    default=5,
    help='''
    Generate sequence with specified length.
    Default is ``5``.
    ''',
    type=int,
  )
  parser.add_argument(
    '--seed',
    default=42,
    help='''
    Random seed.
    Default is ``42``.
    ''',
    type=int,
  )
  parser.add_argument(
    '--n_update',
    default=10,
    help='''
    Number of update times (using Baum-Welch).
    Default is ``10``.
    ''',
    type=int,
  )

  args = parser.parse_args()
  return args


def main(argv: List[str]) -> None:
  # Parse arguments.
  args = parse_args(argv=argv)

  # Set random seed.
  hmm.util.rand.set_seed(seed=args.seed)

  # Load model.
  model = hmm.model.coin.SimpleCoinTossTrials()
  emit_m, init_m, tran_m = model.emit_m, model.init_m, model.tran_m

  # Generate sequences of states and observations.
  sid_seq = hmm.algo.base.gen_sid_seq(
    init_m=init_m,
    seq_len=args.seq_len,
    tran_m=tran_m,
  )
  oid_seq = hmm.algo.base.gen_oid_seq(
    emit_m=emit_m,
    sid_seq=sid_seq,
  )

  print('Generated state sequence:')
  print(model.get_s_seq(sid_seq=sid_seq))
  print('Generated observation sequence conditioned on previously generated state sequence:')
  print(model.get_o_seq(oid_seq=oid_seq))
  print('The joint probability of the observation sequence and state sequence:')
  o_and_s_seq_prob = hmm.algo.base.get_oid_seq_and_sid_seq_prob(
    emit_m=emit_m,
    init_m=init_m,
    oid_seq=oid_seq,
    sid_seq=sid_seq,
    tran_m=tran_m,
  )
  print(o_and_s_seq_prob)

  o_prob = hmm.algo.base.get_oid_seq_prob_by_forward_algo(
    emit_m=emit_m,
    init_m=init_m,
    oid_seq=oid_seq,
    tran_m=tran_m,
  )
  print('Observation probability (by forward algorithm):')
  print(o_prob)

  o_prob = hmm.algo.base.get_oid_seq_prob_by_backward_algo(
    emit_m=emit_m,
    init_m=init_m,
    oid_seq=oid_seq,
    tran_m=tran_m,
  )
  print('Observation probability (by backward algorithm):')
  print(o_prob)

  best_sid_seq = hmm.algo.base.viterbi_algo(
    emit_m=emit_m,
    init_m=init_m,
    oid_seq=oid_seq,
    tran_m=tran_m,
  )
  print('The best state sequence decoded by Viterbi:')
  print(model.get_s_seq(sid_seq=best_sid_seq))

  o_and_s_seq_prob = hmm.algo.base.get_oid_seq_and_sid_seq_prob(
    emit_m=emit_m,
    init_m=init_m,
    oid_seq=oid_seq,
    sid_seq=best_sid_seq,
    tran_m=tran_m,
  )
  print('The joint probability of the observation sequence and the Viterbi decoded state sequence:')
  print(o_and_s_seq_prob)

  ξ1, γ1 = hmm.algo.base.compute_ξ_and_γ(
    emit_m=emit_m,
    init_m=init_m,
    oid_seq=oid_seq,
    tran_m=tran_m,
  )
  ξ2, γ2 = hmm.algo.mat.compute_ξ_and_γ(
    emit_m=emit_m,
    init_m=init_m,
    oid_seq=oid_seq,
    tran_m=tran_m,
  )

  cli_logger = tqdm(range(args.n_update), desc=f'observation probability: {o_prob}')
  for _ in cli_logger:
    emit_m, init_m, tran_m = hmm.algo.mat.baum_welch_algo(
      emit_m=emit_m,
      init_m=init_m,
      oid_seq=oid_seq,
      tran_m=tran_m,
    )
    o_prob = hmm.algo.mat.get_oid_seq_prob_by_forward_algo(
      emit_m=emit_m,
      init_m=init_m,
      oid_seq=oid_seq,
      tran_m=tran_m,
    )
    cli_logger.set_description(f'observation probability: {o_prob}')


if __name__ == '__main__':
  main(argv=sys.argv[1:])
