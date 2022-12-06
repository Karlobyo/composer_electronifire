import streamlit as st
from pygame import mixer
from mido import MidiFile
import time
import numpy as np

mixer.init()

import os

bashCommand = "fluidsynth -ni font.sf2 $test_midi -F $test_midi\.wav -r 16000 > /dev/null"
os.system(bashCommand)
# music = MidiFile('test_midi.mid',clip=True)


# try:
#     mixer.music.load(music)
# except Exception:
#     st.write("Please choose a song")


# audio_file = open('test_midi.mid', 'rb')
# audio_bytes = audio_file.read()

# st.audio(audio_bytes)


# if st.button("Play"):
#     st.audio(music,format='mid')
#     # mixer.music.play()

# if st.button("Stop"):
#     mixer.music.stop()

# if st.button("Resume"):
#     mixer.music.unpause()

# if st.button("Pause"):
#     mixer.music.pause()
