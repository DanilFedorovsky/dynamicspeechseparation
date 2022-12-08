import os
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
with open('/project/data_asr/CHiME5/data/danil/dynamicspeechseparation/mix_2_spk_cv.txt','r') as txt:
    out = txt.readline()
    while out != "":
        out = out.split()
        cv.append([out[0],out[2]])
        out = txt.readline()

i = 1
for file in os.listdir("/project/data_asr/CHiME5/data/wsj0-mix2/twoChannelRoom/cv/mix"):
	i = i+1
	found = False
	for c in cv:
		comparison = c[0].split("/")[-1][:-4] + "_" + c[1].split("/")[-1]
		if file == comparison:
			found=True
	if not found:
		print(file)
print(i)
