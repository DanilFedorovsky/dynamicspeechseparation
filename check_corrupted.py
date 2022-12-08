import wave
import os
from tqdm import tqdm

def is_corrupted(filepath):
    try:
        with wave.open(filepath, 'rb') as wav:
            wav.readframes(wav.getnframes())
        return False
    except (wave.Error, EOFError):
        return True

# check if the file at 'filepath' is corrupted
PATH="/project/data_asr/CHiME5/data/wsj0-mix2/twoChannelRoom/"
subpaths1 = ["tr/","tt/","cv/"]
subpaths2 = ["mix/","s1/","s2/"]
for sub1 in subpaths1:
    for sub2 in subpaths2:
        folder = PATH+sub1+sub2
        for filepath in tqdm(os.listdir(folder)):
            if is_corrupted(folder+filepath):
                print("The file is corrupted:",filepath)