import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile

fs, audio = wavfile.read("./outputs/figs/ex.wav")

# WSO 2 MIX ROOM SETUP -> CHECK PAPER -> COPY ROOM SETUP (LIST OF ROOMS)
# Create a 4 by 6 metres shoe box room
room = pra.ShoeBox([4,6]) # Or AnechoicRoom 

# Add a source somewhere in the room
room.add_source([2.5, 4.5], signal=audio)

# Create a linear array with 2 microphones
# with angle 0 degrees and inter mic distance 10 cm
R = pra.linear_2D_array([2, 1.5], 2, 0, 0.1)
room.add_microphone_array(pra.Beamformer(R, room.fs))

# Simulate propagation
room.simulate()

room.mic_array.to_wav(
    "./outputs/figs/ex_out.wav",
    norm=True,
    bitdepth=np.int16,
)
