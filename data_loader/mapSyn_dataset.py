import argparse
from pathlib import Path
import random
import math

import librosa
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data.preprocessor import SAMPLING_RATE
from data import (
    Zscore, ExtractSpectrogram, LogCompress, Clipping, MinMaxNorm, DimMod
)


class MAPSynth(Dataset):
    def __init__(self, datasets_path, seq_len='min', rand_seg=True):
        self.datasets_path = list(datasets_path)
        self.wav_path = sorted(self._gather_file(self.datasets_path, 'wav'))

        self.n_files = len(self.wav_path)
        msg = "No WAV files found, run `mod_synth_midi.py` first."
        # TODO: abstract `mod_synth_midi.py` and enable execution here without
        # spiting the assertion error
        assert self.n_files > 0, msg

        self.pt_path = sorted(self._gather_file(self.datasets_path, 'pt'))
        self.pt_files = self._load_pt(self.pt_path)

        self.min_duration = self._check_min_dur(self.pt_files)
        if seq_len != 'min':
            msg = "`seq_len` should be shorter than the minimum duration."
            assert seq_len <= self.min_duration, msg
            self.seq_len = seq_len
        else:
            self.seq_len = self.min_duration
        self.rand_seg = rand_seg

        self.transform = transforms.Compose([
            Zscore(),
            ExtractSpectrogram(),
            LogCompress(),
            Clipping(),
            MinMaxNorm(),
        ])
   
    def _gather_file(self, datasets_path, ext):
        audio_path = []
        for d in datasets_path:
            audio_path.append(list(Path(d).glob(f'*.{ext}')))
        return [k for f in audio_path for k in f]

    def _load_pt(self, file_path):
        if len(file_path) != self.n_files:
            print((
                f"Unequal numbers of PT ({len(file_path)}) " 
                f"and WAV ({self.n_files}) files."
            ))
            print("Creating and saving PT files ...")
            return self._load_wav()
        else:
            try:
                return [torch.load(f) for f in file_path]
            except RuntimeError:
                return [torch.jit.load(f) for f in file_path]
    
    def _load_wav(self):
        audio_pt = []
        for f in self.wav_path:
            par_dir = Path(Path(f).parents[0])
            audio, _ = librosa.load(f, sr=SAMPLING_RATE)
            audio = torch.from_numpy(audio)
            torch.save(audio, par_dir / Path(Path(f).stem + '.pt'))
            audio_pt.append(audio)
        return audio_pt

    def _check_min_dur(self, pt_files):
        min_dur = math.inf
        for f in pt_files:
            dur = len(f) / SAMPLING_RATE
            if dur < min_dur:
                min_dur = dur 
        return min_dur

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        audio = self.pt_files[idx]
        if self.rand_seg:
            start = random.randint(
                0, len(audio) - int(self.seq_len * SAMPLING_RATE)
            )
        else:
            start = 0
        end = start + int(self.seq_len * SAMPLING_RATE)
        S = self.transform(audio[start:end]).squeeze(0)

        return idx, S.transpose(0, -1), S.size(-1)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--datasets', nargs='+')
    args = args.parse_args()

    datasets = args.datasets
    d = MAPSynth(datasets)
    print((
        f"{len(d.pt_files)} pt files are found from {datasets}.\n"
        f"Minimum audio duration (sec.) is {d.min_duration}."
    ))
    print(d[5])