import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
import os
from tqdm import tqdm

PATH = "/project/data_asr/CHiME5/data/wsj0-mix2/2speakers/wav8k/min/tr/mix/"

for file in tqdm(os.listdir(PATH)):
    # TWO / 3 SIGNAL SOURCES!!!

    fs, audio = wavfile.read(PATH+file)

    # Create a 4 by 6 metres shoe box room
    room = pra.ShoeBox([4,6]) # Or AnechoicRoom 
    room.add_source([2.5, 4.5], signal=audio)

    # Create a linear array with 2 microphones with angle 0 degrees and inter mic distance 10 cm
    R = pra.linear_2D_array([2, 1.5], 1, 0, 0.1)
    room.add_microphone_array(pra.Beamformer(R, room.fs))

    # Simulate propagation
    room.simulate()

    room.mic_array.to_wav(
        ("/project/data_asr/CHiME5/data/wsj0-mix2/oneChannelRoom/tr/mix/" + str(file)),
        norm=True,
        bitdepth=np.int16,
    )
