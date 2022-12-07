import wave
import os
from tqdm import tqdm
from scipy.io import wavfile

CONV_PATH = "/project/data_asr/CHiME5/data/wsj0-mix2/"
channels = ["twoChannelRoom/", "oneChannelRoom/"]
datasets = ["cv/","tr/","tt/"]
scenarios = ["mix/","s1/","s2/"]

for channel in channels:
    for dataset in datasets:
        for scenario in scenarios:
            for file in tqdm(os.listdir(CONV_PATH+channel+dataset+scenario)):
                file_path = os.path.join(CONV_PATH, channel, dataset, scenario, file)
                print(file_path)
                data, sample_rate = wavfile.read(file_path)
                print(sample_rate)
                wavfile.write(file_path, data, 8000)
