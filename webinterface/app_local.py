#from fastapi import FastAPI
import streamlit as st
#import numpy as np
#from PIL import Image
import base64
#import time
#import os
#import requests
#import wavfile
#import pretty_midi
#import io


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

st.title('Composer Electronifire')

st.markdown('### *Electronifire the big maestros from the past...*')

st.markdown("")

st.markdown("")

st.image('images/question_pic.png',width=50)

st.markdown("Have you ever asked yourself...")
st.markdown("How would that great composer from the past sound like, if he/she would be alive?")

st.image('images/question_man.png',width=50)


st.markdown("")

st.markdown('The opinion of ChatGPT:')
st.markdown("It's impossible to say exactly how a composer from the past would sound like if they were alive today. Their musical style would likely be influenced by the current trends, as well as their own personal experiences. Additionally, it's worth considering that the tools and technology available for making music have changed dramatically since the composer's time, so their approach to composition might be quite different as well. Ultimately, it's a hypothetical question that can't be answered with certainty, but it's certainly an interesting experiment!")

st.markdown("")

st.markdown("Enjoy!")

st.markdown("(Demo version: only Chopin available at the moment)")

st.markdown("")

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

        if  st.button('Electroni🔥',on_click=callback):

            audio_file = open('../electronifired_music/chopin_1_TRIPHOP.wav', 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')

            with open('../electronifired_music/chopin_1_TRIPHOP.wav', "rb") as file:
                btn = st.download_button(
                label="Download Electronifired music",
                data=file,
                file_name= f"chopin_1_triphop.wav",
                mime="wav/wav"
                )


            add_bg_from_local('images/Electronifire_pic.jpg')



            #Trip-hop == '../electronifired_music/chopin_1_TRIPHOP.wav'
            #Techno == '../electronifired_music/chopin_1_TECHNO.wav'
            #ElecMusic = ['Trip-hop', 'Techno']

            #selec_options = st.multiselect(label='Choose a beat to electroni-🔥 your tune',options=ElecMusic, on_change = None,key=1)
            #on_change will call the second API with the argument that is chosen by user


            #if selec_options:
               # for option in selec_options:
                   # if option == 'Trip-hop':
                   #     file_path = '../electronifired_music/chopin_1_TRIPHOP.wav'

                   # elif option == 'Techno':
                   #     file_path = '../electronifired_music/chopin_1_TECHNO.wav'

#
                   # audio_file = open(file_path, 'rb')
                   # audio_bytes = audio_file.read()
                   # st.audio(audio_bytes, format='audio/wav')
#
                   # with open(file_path, "rb") as file:
                   #     btn = st.download_button(
                   #     label="Download Electronifired music",
                   # #    data=file,
                   #     file_name= f"chopin_1_{option}.wav",
                   #     mime="wav/wav"
                   #     )




                  #  add_bg_from_local('images/discoball2.jpg')



if st.checkbox('Claude Debussy',value=False,args=st.image('images/debussy.jpeg',width=100)):
     st.write('Good choice, but how about Chopin?')
