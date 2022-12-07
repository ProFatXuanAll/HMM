import src.dset.coin
import src.hmm.first_order
import src.util.rand
import argparse
import sys
from typing import List


def parse_args(argv: List[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser('python -m src.script.demo_simple_coin')
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

  dset = src.dset.coin.SimpleCoinTossTrials()
  model = src.hmm.first_order.FirstOrderHMM(emit_m=dset.emit_m, init_m=dset.init_m, tran_m=dset.tran_m)

  state_id_seq, obs_id_seq = model.gen_seq(seq_len=10)
  print(dset.get_obs(obs_id_seq=obs_id_seq))
  print(dset.get_state(state_id_seq=state_id_seq))


if __name__ == '__main__':
  main(argv=sys.argv[1:])