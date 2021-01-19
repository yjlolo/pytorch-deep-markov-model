import argparse

from torchvision import transforms

from base import BaseDataLoader
import data_loader.polyphonic_dataset as poly
from data_loader.mapSyn_dataset import MAPSynth
from data_loader.seq_util import seq_collate_fn
from data import (
    Zscore, ExtractSpectrogram, LogCompress, Clipping, MinMaxNorm
)


class PolyMusicDataLoader(BaseDataLoader):
    def __init__(self,
                 batch_size,
                 data_dir='jsb',
                 split='train',
                 shuffle=True,
                 collate_fn=seq_collate_fn,
                 num_workers=1):

        assert data_dir in ['jsb']
        assert split in ['train', 'valid', 'test']
        if data_dir == 'jsb':
            self.dataset = poly.PolyDataset(poly.JSB_CHORALES, split)
        self.data_dir = data_dir
        self.split = split

        super().__init__(self.dataset,
                         batch_size,
                         shuffle,
                         0.0,
                         num_workers,
                         seq_collate_fn)


class MAPSynthDataLoader(BaseDataLoader):
    def __init__(
        self,
        batch_size,
        datasets_path,
        seq_len='min',
        shuffle=True,
        validation_split=0.0,
        num_workers=1
    ):
        trsfm = transforms.Compose([
            Zscore(),
            ExtractSpectrogram(),
            LogCompress(),
            Clipping(),
            MinMaxNorm()
        ])
        self.dataset = MAPSynth(datasets_path, seq_len, trsfm)

        super().__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers
        )


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--datasets', nargs='+')
    args = args.parse_args()

    dl = MAPSynthDataLoader(10, args.datasets, seq_len=10)
    print(next(iter(dl)).size())
