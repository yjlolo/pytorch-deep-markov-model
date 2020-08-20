from base import BaseDataLoader
import data_loader.polyphonic_dataloader as poly
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
                         seq_collate_fn)
