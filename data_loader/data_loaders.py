import argparse

from base import BaseDataLoader
import data_loader.polyphonic_dataset as poly
from data_loader.mapSyn_dataset import MAPSynth
from data_loader.seq_util import seq_collate_fn


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
                         collate_fn)


class MAPSynthDataLoader(BaseDataLoader):
    def __init__(
        self,
        batch_size,
        datasets_path,
        seq_len='min',
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        collate_fn=seq_collate_fn
    ):
        self.dataset = MAPSynth(datasets_path, seq_len)

        super().__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            collate_fn
        )


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--datasets', nargs='+')
    args = args.parse_args()

    dl = MAPSynthDataLoader(10, args.datasets, seq_len=10)
    x_batch, x_rev_batch, mask, x_len = next(iter(dl))
    print(x_batch.size(), x_rev_batch.size(), mask.size(), x_len.size())
