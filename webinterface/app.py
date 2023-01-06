from fastapi import FastAPI
import streamlit as st
import numpy as np
from PIL import Image
import base64
import time
import os
import requests
import wavfile
import pretty_midi
import io


if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False
def callback():
    st.session_state.button_clicked = True


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )



# app = FastAPI()

# # Define a root `/` endpoint
# @app.get('/')
# def index():
#     return {'ok': True}

# @app.get('/electronify')

st.set_page_config(layout='wide')

CSS = """
h1 {
    color: white;
}
h2 {color: black;
}
.stApp {
    background-image: url(https://cdn.pixabay.com/photo/2013/12/25/18/15/piano-233715_1280.jpg);
    background-size: cover;
"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

st.title(':collision: Composer Electronifire :collision:')

st.markdown('Please select a composer:')


if st.checkbox('Johann Sebastian Bach',value=False,args=st.image('images/bach.jpeg',width=100)):
     st.write('Good choice, but how about Chopin?')

if st.checkbox('Franz Schubert',value=False,args=st.image('images/schubert.jpeg',width=100)):
     st.write('Good choice, but how about Chopin?')

if st.checkbox('Frédéric Chopin',value=False,args=st.image('images/Freddy.jpeg',width=100)):
    if (st.button('Generate new music',on_click=callback) or st.session_state.button_clicked):

        audio_file = open('../composer_music/chopin_1.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

        with open('../composer_music/chopin_1.wav', "rb") as file:
            btn = st.download_button(
            label="Download Composer music",
            data=file,
            file_name="chopin_1.wav",
            mime="wav/wav"
            )

        if  st.button('Electroni-🔥',on_click=callback):
            ElecMusic = ['../chopin_1_TRIPHOP.wav', '../chopin_1_TECHNO.wav']

            selec_audios = st.multiselect(label='Choose a beat to electroni-🔥 your tune',options=ElecMusic, on_change = None,key=1)
            #on_change will call the second API with the argument that is chosen by user

            for audio in selec_audios:
                audio_file = open(audio, 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/wav')

                add_bg_from_local('images/CompElecPic.png')

            with open("../electronifired_music/chopin_1_TRIPHOP.wav", "rb") as file:
                btn = st.download_button(
                label="Download Composer Electronifire music",
                data=file,
                file_name="chopin_1_TRIPHOP.wav",
                mime="wav/wav")


if st.checkbox('Claude Debussy',value=False,args=st.image('images/debussy.jpeg',width=100)):
     st.write('Good choice, but how about Chopin?')
