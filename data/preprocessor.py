import sys
import math
import numpy as np

import torch
from torch import nn

from nnAudio.Spectrogram import MelSpectrogram


SAMPLING_RATE = 16000
NFFT = 1024 
HOP = 128 
NMEL = 80
CHUNK_DUR = 0.5
CHUNK_SIZE = int(SAMPLING_RATE * CHUNK_DUR) // HOP

use_cuda = True if torch.cuda.is_available() else False


class Zscore():
    def __init__(self, divide_sigma=False):
        self.divide_sigma = divide_sigma

    def __call__(self, x):
        x -= x.mean()
        if self.divide_sigma:
            x /= x.std()
        return x


class LogCompress():
    def __call__(self, x):
        return torch.log(sys.float_info.epsilon + x)


class Clipping():
    def __init__(self, clip_min=-100, clip_max=100):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, x):
        x[x <= self.clip_min] = self.clip_min
        # x[x >= self.clip_max] = self.clip_max
        return x


class MinMaxNorm:
    def __init__(self, min_val=-1, max_val=1, x_min=None, x_max=None):
        self.min_val = min_val
        self.max_val = max_val
        self.x_min = x_min
        self.x_max = x_max

    def __call__(self, x):
        x_min = x.min() if self.x_min is None else self.x_min
        x_max = x.max() if self.x_max is None else self.x_max

        nom = x - x_min
        den = x_max - x_min

        if abs(den) > 1e-4:
            return (self.max_val - self.min_val) * (nom / den) + self.min_val
        else:
            return nom


class ExtractSpectrogram():
    def __init__(
        self, sr=SAMPLING_RATE, n_fft=NFFT, hop_length=HOP, n_mels=NMEL
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.spectrogram = MelSpectrogram(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length
        )

    def __call__(self, x):
        return self.spectrogram(x.float())


class DimMod():
    def __init__(self, op, dim):
        assert op in ['squeeze', 'unsqueeze']
        self.op = op
        self.dim = dim
    
    def __call__(self, x):
        if self.op == 'squeeze':
            return x.squeeze(self.dim)
        return x.unsqueeze(self.dim)


class SpecChunking():
    def __init__(
        self,
        duration=CHUNK_DUR,
        sr=SAMPLING_RATE,
        hop_length=HOP,
        only_first_seg=False
    ):
        """
        Slice spectrogram into non-overlapping chunks. Discard chunks shorter than the specified duration.
        :params duration: the duration (in sec.) of each spectrogram chunk
        :params sr: sampling frequency used to read waveform; used to calculate the chunk size
        :params hop_length: hop size used to derive spectrogram; used to calculate the chunk size
        """
        self.duration = duration
        self.sr = sr
        self.hop_length = hop_length
        self.chunk_size = int(sr * duration) // hop_length
        self.only_first_seg = only_first_seg
        #self.overlap_size = int(self.chunk_size * overlap)
        #self.reverse = reverse

    def __call__(self, x):
        time_dim = -1  # assume input spectrogram with shape (freq, time) or (batch, freq, time)

        y = torch.split(x, self.chunk_size, dim=time_dim)[:-1]
        y = torch.cat(y, dim=0)

        if self.only_first_seg:
            y = y[0].unsqueeze(0)

        return y