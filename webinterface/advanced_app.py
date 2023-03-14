from fastapi import FastAPI
import streamlit as st
import numpy as np
from PIL import Image
import io
from mido import MidiFile, open_output



st.set_page_config(layout='wide')

CSS = """
h1 {
    color: white;
}
h2 {color: black;
}
.stApp {
    background-image: url(https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Chateau_de_Versailles_1668_Pierre_Patel.jpg/1920px-Chateau_de_Versailles_1668_Pierre_Patel.jpg);
    background-size: cover;
"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)


st.title(':collision::collision: Advanced Composer Electronifire :collision::collision:')

st.markdown('Please select an MC:')


if st.checkbox('Johann Sebastian Bach',st.image('bach.jpeg',width=100)):
     st.write('Good choice, but how about Chopin?')

if st.checkbox('Franz Schubert',st.image('schubert.jpeg',width=100)):
     st.write('Good choice, but how about Chopin?')

if st.checkbox('Frédéric Chopin',st.image('Freddy.jpeg',width=100)):
     if st.button('Electroni-🔥'):
         CSS = """
                h1 {
                color: white;
                }
            .stApp {
            background-image: url(https://upload.wikimedia.org/wikipedia/commons/b/b1/Techno-Club_About_Blank_Markgrafendamm_Berlin.jpg);
            background-size: cover;
            """
         st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

         import time

           # Add a placeholder
         latest_iteration = st.empty()
         bar = st.progress(0)

         for i in range(100):
            # Update the progress bar with each iteration.
            if 0<=i<33:
                latest_iteration.text(f'Getting matches')
            if 34<=i<66:
                latest_iteration.text(f'Making fire')
            if 66<=i<98:
                latest_iteration.text(f"It's getting hotter")
            if i==99:
                latest_iteration.text(f"🔥Burn the house down!🔥")


            bar.progress(i + 1)
            time.sleep(0.01)



         st.download_button('Download your midi file',data='test_midi.mid',file_name='test_midi.mid')



        #  with st.spinner(f"Transcribing to FluidSynth"):
        #   midi_data = pretty_midi.PrettyMIDI('test_midi.mid')
        #   audio_data = midi_data.fluidsynth()
        #   audio_data = np.int16(
        #     audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
        #   )  # -- Normalize for 16 bit audio https://github.com/jkanner/streamlit-audio/blob/main/helper.py

        #   virtualfile = io.BytesIO()
        #   wavfile.write(virtualfile, 44100, audio_data)

        #   st.audio(virtualfile)
        #   st.markdown("Download the audio by right-clicking on the media player")

if st.checkbox('Claude Debussy',st.image('debussy.jpeg',width=100)):
     st.write('Good choice, but how about Chopin?')
