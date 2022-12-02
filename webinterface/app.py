from fastapi import FastAPI
import streamlit as st
import numpy as np
from PIL import Image
import base64
import time
import os
import requests
import fluidsynth
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
    background-image: url(https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Chateau_de_Versailles_1668_Pierre_Patel.jpg/1920px-Chateau_de_Versailles_1668_Pierre_Patel.jpg);
    background-size: cover;
"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

st.title(':collision: Composer Electronifire :collision:')

st.markdown('Please select an MC:')


if st.checkbox('Johann Sebastian Bach',value=False,args=st.image('bach.jpeg',width=100)):
     st.write('Good choice, but how about Chopin?')

if st.checkbox('Franz Schubert',value=False,args=st.image('schubert.jpeg',width=100)):
     st.write('Good choice, but how about Chopin?')

if st.checkbox('Frédéric Chopin',value=False,args=st.image('Freddy.jpeg',width=100)):
     if (st.button('Generate a new tune',on_click=callback) or st.session_state.button_clicked):
         add_bg_from_local('discoball2.jpg')
         """1st API call here to generate a new tune"""

         with open("test_tune1.mid", "rb") as file:
            btn = st.download_button(
            label="Download tune",
            data=file,
            file_name="test_tune1.mid",
            mime="mid/mid"
          )
            add_bg_from_local('discoball2.jpg')

         st.multiselect(label='Choose a beat to electroni-🔥 your tune.',options=['bebop','funk','hardcore techno'], on_change = None,key=1)
         #on_change will call the second API with the argument that is chosen by user
         """2nd API call here to generate tune with beat"""

         with open("test_tune2.mid", "rb") as file:
            btn = st.download_button(
            label="Download tune",
            data=file,
            file_name="test_tune2.mid",
            mime="mid/mid")


if st.checkbox('Claude Debussy',value=False,args=st.image('debussy.jpeg',width=100)):
     st.write('Good choice, but how about Chopin?')