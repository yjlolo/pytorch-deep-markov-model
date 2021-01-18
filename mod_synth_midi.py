import random
from pathlib import Path
import argparse

import pretty_midi
from pretty_midi import constants
from midi2audio import FluidSynth
from utils import filter_with_ext, ensure_dir


FILE_EXT = 'mid'
INSTRUMENT_MAP = {
    'ap': 'Acoustic Grand Piano',
    'ep': 'Electric Grand Piano',
    'co': 'Church Organ',
    'guitar': 'Acoustic Guitar (nylon)',
    'ps': 'Pizzicato Strings',
    'os': 'Orchestral Harp',
    'se': 'String Ensemble 1',
    'sl': 'Lead 1 (square)'
}
SAMPLE_RATE = 16000


def main(args):
    filepath = args.filepath
    target_instruments = args.instruments
    n_rand = args.n_rand

    midi_filepath = filter_with_ext(filepath, FILE_EXT)
    print(f"Identified {len(midi_filepath)} {FILE_EXT} files.")
    if n_rand != -1:
        random.seed(888)
        midi_filepath_sampled = random.sample(midi_filepath, k=n_rand)
    else:
        midi_filepath_sampled = midi_filepath
    print((
        f"{len(set(midi_filepath_sampled))}/{len(set(midi_filepath))} "
        f"{FILE_EXT} files are read."
    ))

    for f in midi_filepath_sampled:
        midi_data = pretty_midi.PrettyMIDI(str(f))
    
        for t in target_instruments:
            try:
                instrument_id = INSTRUMENT_MAP[t]
            except KeyError:
                t = t.lower()
                assert t in ['vibraphone', 'trumpet']
                instrument_id = t.title()
            program_id = constants.INSTRUMENT_MAP.index(instrument_id)
            for i in range(len(midi_data.instruments)):
                midi_data.instruments[i].program = program_id
    
            source_filename = Path(f).stem
            prefix = ''.join(instrument_id.split())
            target_filename = Path(f'{source_filename}-{prefix}.{FILE_EXT}')
            target_dataset = Path(f'{Path(f).parents[0]}-{prefix}')
            ensure_dir(target_dataset)
            output_path = target_dataset / target_filename
            midi_data.write(str(output_path))
            print(f"{target_filename} has been saved to {output_path}.")

            wav_output = target_dataset / Path(target_filename.stem + '.wav')
            fs = FluidSynth(
                sample_rate=SAMPLE_RATE, 
                sound_font='constants/MuseScore_General.sf3'
            )
            fs.midi_to_audio(str(output_path), str(wav_output))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-f', '--filepath', type=str)
    args.add_argument('-i', '--instruments', nargs='+')
    args.add_argument('-n', '--n_rand', type=int, default=-1)

    args = args.parse_args()
    main(args)
