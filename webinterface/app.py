from fastapi import FastAPI
import streamlit as st
import numpy as np
from PIL import Image
import base64
import time



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
     if st.button('Generate a new tune'):
         add_bg_from_local('discoball2.jpg')
         latest_iteration = st.empty()
         bar = st.progress(0)

         for i in range(100):
            # Update the progress bar with each iteration.
            latest_iteration.text("🎶New tune is being generated🎶")
            bar.progress(i + 1)
            time.sleep(0.05)

         if st.download_button('Listen to your new tune',data='test_midi.mid', key=0):
            #st.cache(persist=True)
            add_bg_from_local('discoball2.jpg')
            if st.button('Electroni-🔥'):
                add_bg_from_local('aboutblank.jpg')
                latest_iteration = st.empty()
                bar = st.progress(0)
                for j in range(100):
                    # Update the progress bar with each iteration.
                    if 0<=j<33:
                        latest_iteration.text(f'Getting matches')
                    if 34<=j<66:
                        latest_iteration.text(f'Making fire')
                    if 66<=j<98:
                        latest_iteration.text(f"It's getting hotter")
                    if j==99:
                        latest_iteration.text(f"🔥Burn the house down!🔥")
                    bar.progress(j + 1)
                    time.sleep(0.01)

                    st.download_button('Download your midi file',data='test_midi.mid',file_name='electronifired_tune.mid',key=1)


if st.checkbox('Claude Debussy',value=False,args=st.image('debussy.jpeg',width=100)):
     st.write('Good choice, but how about Chopin?')
