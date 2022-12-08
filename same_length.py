import os
import torchaudio
import torch
from tqdm import tqdm

PATH = "/project/data_asr/CHiME5/data/wsj0-mix2/oneChannelRoom/"
parts = ["cv/","tr/","tt/"]
scenarios = ["mix/","s1/","s2/"]

def same_length(a:torch.Tensor,b:torch.Tensor):
    len_a = a.shape[1]
    len_b = b.shape[1]  
    if len_a > len_b:
        a = a.narrow(1,0,len_b)
    elif len_b > len_a:
        b = b.narrow(1,0,len_a)
    return a, b

def same_length3(a:torch.Tensor,b:torch.Tensor,c:torch.Tensor):
    a,b = same_length(a,b)
    b,c = same_length(b,c)
    a,c = same_length(a,c)
    return a,b,c

for part in parts:
    for filename in tqdm(os.listdir(PATH+part+scenarios[0])):

        a, sr = torchaudio.load(PATH+part+scenarios[0]+filename)
        b, _ = torchaudio.load(PATH+part+scenarios[1]+filename)
        c, _ = torchaudio.load(PATH+part+scenarios[2]+filename)

        a,b,c = same_length3(a,b,c)
        
        torchaudio.save(PATH+part+scenarios[0]+filename, a, sr)
        torchaudio.save(PATH+part+scenarios[1]+filename, b, sr)
        torchaudio.save(PATH+part+scenarios[2]+filename, c, sr)

