import argparse
import sys
from typing import List

import src.dset.coin
import src.hmm.first_order
import src.util.rand


def parse_args(argv: List[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser('python -m src.script.demo_simple_coin')
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

  args = parser.parse_args()
  return args


def main(argv: List[str]) -> None:
  # Parse arguments.
  args = parse_args(argv=argv)

  # Set random seed.
  src.util.rand.set_seed(seed=args.seed)

  # Load dataset.
  dset = src.dset.coin.SimpleCoinTossTrials()

  # Load model.
  model = src.hmm.first_order.FirstOrderHMM(emit_m=dset.emit_m, init_m=dset.init_m, tran_m=dset.tran_m)

  state_id_seq, obs_id_seq = model.gen_seq(seq_len=args.seq_len)
  print('Generated observation:')
  print(dset.get_obs(obs_id_seq=obs_id_seq))
  print('Corresponding hidden states:')
  print(dset.get_state(state_id_seq=state_id_seq))

  obs_prob, _ = model.forward_algo(obs_id_seq=obs_id_seq)
  print('Observation probability (by forward algorithm):')
  print(obs_prob)

  obs_prob, _ = model.backward_algo(obs_id_seq=obs_id_seq)
  print('Observation probability (by backward algorithm):')
  print(obs_prob)

  best_obs_n_state_prob, best_state_id_seq = model.viterbi_algo(obs_id_seq=obs_id_seq)
  print('The best joint probability on observation and state:')
  print(best_obs_n_state_prob)
  print('The best state sequence decoded by Viterbi:')
  print(dset.get_state(state_id_seq=best_state_id_seq))


if __name__ == '__main__':
  main(argv=sys.argv[1:])
