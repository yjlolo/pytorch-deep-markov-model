from pathlib import Path
import argparse

from music21 import converter, instrument 
from utils import filter_with_ext, ensure_dir


FILE_EXT = 'mid'
SOURCE_INSTRUMENT = 'Piano'

def main(args):
    filepath = args.filepath
    target_instruments = args.instruments

    midi_filepath = filter_with_ext(filepath, FILE_EXT)
    print(f"Identified {len(midi_filepath)} {FILE_EXT} files.")
    target_instruments = [t.lower().title() for t in target_instruments]
    for t in target_instruments:
        assert hasattr(instrument, t), "Invalid target instrument: %s." % t
    
    for f in midi_filepath:
        s = converter.parse(f)
    
        for t in target_instruments:
            for el in s.recurse():
                if SOURCE_INSTRUMENT in el.classes: 
                    el.activeSite.replace(el, getattr(instrument, t)())
    
            source_filename = Path(f).stem
            target_filename = f'{source_filename}-to{t}.{FILE_EXT}'
            target_dataset = f'{Path(f).parents[0]}-to{t}'
            ensure_dir(target_dataset)
            output_path = Path(target_dataset) / target_filename
            s.write('midi', output_path)
            print(f"{target_filename} has been saved to {output_path}.")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-f', '--filepath', type=str)
    args.add_argument('-i', '--instruments', nargs='+')

    args = args.parse_args()
    main(args)
