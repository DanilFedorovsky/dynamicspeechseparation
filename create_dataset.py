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
        #cv.append([out[0],out[2]])
        cv.append(out)
        out = txt.readline()

tr = []
with open('mix_2_spk_tr.txt','r') as txt:
    out = txt.readline()
    while out != "":
        out = out.split()
        #tr.append([out[0],out[2]])
        tr.append(out)
        out = txt.readline()

tt = []
with open('mix_2_spk_tt.txt','r') as txt:
    out = txt.readline()
    while out != "":
        out = out.split()
        #tt.append([out[0],out[2]])
        tt.append(out)
        out = txt.readline()

print("tr:",len(tr))
print("cv:",len(cv))
print("tt:",len(tt))
print(tr[0])

def same_length(a:torch.Tensor,b:torch.Tensor):
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    len_a = a.shape[0]
    len_b = b.shape[0]  
    if len_a > len_b:
        #add = len_a - len_b
        #b = torch.cat([b,torch.zeros(add)],dim=0)
        a = a.narrow(0,0,len_b)
    elif len_b > len_a:
        #add = len_b - len_a
        #a = torch.cat([a,torch.zeros(add)],dim=0)
        # Narrow b
        b = b.narrow(0,0,len_a)
    return a, b

channels = ["twoChannelRoom/", "oneChannelRoom/"]
scenarios = ["mix/","s1/","s2/"]

def create_dataset(part: list, subfolder: str):
    for channel in channels:
        CREATE_PATH = "/project/data_asr/CHiME5/data/wsj0-mix2/" + channel
        os.makedirs(CREATE_PATH,exist_ok=True)
        os.makedirs(CREATE_PATH+subfolder,exist_ok=True)
        for scenario in scenarios:
            os.makedirs(CREATE_PATH+subfolder+scenario, exist_ok=True)
            print("Generate",scenario,"for",subfolder)
            for line in tqdm(part):
                # Load two audiofiles
                
                fs, wav_a = wavfile.read(wsj0path+line[0])
                fs, wav_b = wavfile.read(wsj0path+line[2])
                wav_a, wav_b = same_length(wav_a,wav_b)
                wav_a = wav_a.numpy()
                wav_b = wav_b.numpy()
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

                name_file = line[0].split("/")[-1][:-4] + "_" + line[1] + "_" + line[2].split("/")[-1][:-4] + line[3] + ".wav"
                room.mic_array.to_wav(
                    (CREATE_PATH + subfolder + scenario + name_file),
                    8000,
                    norm=True,
                    bitdepth=np.int16,
                )
    return "Done"

create_dataset(cv,"cv/")
create_dataset(tr,"tr/")
create_dataset(tt,"tt/")