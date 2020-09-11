
# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Tuple

class NERDataset:
    r"""Name-entity recognition dataset."""

    def __init__(self):
        self.samples = (
            (
                'Kevin Chou is a handsome guy .',
                ('B', 'I', 'O', 'O', 'O', 'O', 'O'),
            ),
            (
                'Kevin Chou like to go to Chinese restaurant .',
                ('B', 'I', 'O', 'O', 'O', 'O', 'B', 'O', 'O'),
            ),
            (
                'But Chinese restaurant do not welcome handsome guys .',
                ('O', 'B', 'O', 'O', 'O', 'O', 'O', 'O', 'O'),
            ),
            (
                'So Kevin Chou go to Japanese restaurant instead of Chinese restaurant .',
                ('O', 'B', 'I', 'O', 'O', 'B', 'O', 'O', 'O', 'B', 'O', 'O'),
            ),
        )

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> Tuple[str, Tuple[str]]:
        return self.samples[index]

class POSDataset:
    r"""Part-of-speech dataset."""

    def __init__(self):
        self.samples = (
            (
                'Kevin Chou is a handsome guy .',
                ('NNP', 'NNP', 'VBZ', 'DT', 'JJ', 'NN', '.'),
            ),
            (
                'Kevin Chou like to go to Chinese restaurant .',
                ('NNP', 'NNP', 'IN', 'TO', 'VB', 'TO', 'NNP', 'NN', '.'),
            ),
            (
                'But Chinese restaurant do not welcome handsome guys .',
                ('CC', 'NNP', 'NN', 'VBP', 'RB', 'VB', 'JJ', 'NNS', '.'),
            ),
            (
                'So Kevin Chou go to Japanese restaurant instead of Chinese restaurant .',
                ('RB', 'NNP', 'NNP', 'VB', 'TO', 'NNP', 'NN', 'RB', 'IN', 'NNP', 'NN', '.'),
            ),
        )

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> Tuple[str, Tuple[str]]:
        return self.samples[index]
