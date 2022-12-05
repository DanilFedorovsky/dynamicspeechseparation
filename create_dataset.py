import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
import os
from tqdm import tqdm
import torchaudio
import torch
from scipy.io import wavfile

wsj0path = "/project/data_asr/wham_dataset/whamr_data/wsj0_raw/"

cv = []
with open('mix_2_spk_cv.txt','r') as txt:
    out = txt.readline()
    while out != "":
        out = out.split()
        cv.append([out[0],out[2]])
        out = txt.readline()

tr = []
with open('mix_2_spk_tr.txt','r') as txt:
    out = txt.readline()
    while out != "":
        out = out.split()
        tr.append([out[0],out[2]])
        out = txt.readline()

tt = []
with open('mix_2_spk_tt.txt','r') as txt:
    out = txt.readline()
    while out != "":
        out = out.split()
        tt.append([out[0],out[2]])
        out = txt.readline()

print("tr:",len(tr))
print("cv:",len(cv))
print("tt:",len(tt))
print(tr[0])

def create_dataset(part: list, subfolder: str):
    channels = ["twoChannelRoom/", "oneChannelRoom/"]
    for channel in channels:
        CREATE_PATH = "/project/data_asr/CHiME5/data/wsj0-mix2/" + channel
        os.mkdir(CREATE_PATH)
        os.mkdir(CREATE_PATH+subfolder)
        scenarios = ["mix/","s1/","s2/"]
        for scenario in scenarios:
            os.mkdir(CREATE_PATH+subfolder+scenario)
            print("Generate",scenario,"for",subfolder)
            for line in tqdm(part):
                # Load two audiofiles
                
                fs, wav_a = wavfile.read(wsj0path+line[0])
                fs, wav_b = wavfile.read(wsj0path+line[1])
                # Simulate room and obtain mix / s1 / s2
                room = pra.ShoeBox([4,6], fs=fs)

                if scenario == "mix/":
                    room.add_source([2.5, 3.5], signal=wav_a, delay=0)
                    room.add_source([0.5, 3.0], signal=wav_b, delay=0)
                    if channel == "oneChannelRoom/":
                        R = pra.linear_2D_array([2, 1.5], 1, 0, 0.1)
                        room.add_microphone_array(pra.Beamformer(R, room.fs))
                    elif channel == "twoChannelRoom/":
                        R = pra.linear_2D_array([2, 1.5], 2, 0, 0.1)
                        room.add_microphone_array(pra.Beamformer(R, room.fs))
                elif scenario == "s1/":
                    room.add_source([2.5, 3.5], signal=wav_a, delay=0)
                    R = pra.linear_2D_array([2, 1.5], 1, 0, 0.1)
                    room.add_microphone_array(pra.Beamformer(R, room.fs))
                elif scenario == "s2/":
                    room.add_source([0.5, 3.0], signal=wav_b, delay=0)
                    R = pra.linear_2D_array([2, 1.5], 1, 0, 0.1)
                    room.add_microphone_array(pra.Beamformer(R, room.fs))
                
                room.simulate()

                name_file = line[0].split("/")[-1][:-4] + "_" + line[1].split("/")[-1]
                room.mic_array.to_wav(
                    (CREATE_PATH + subfolder + scenario + name_file),
                    norm=True,
                    bitdepth=np.int16,
                )
    return "Done"

create_dataset(cv,"cv/")
create_dataset(tr,"tr/")
create_dataset(tt,"tt/")