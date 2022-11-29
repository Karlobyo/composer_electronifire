from fastapi import FastAPI
import streamlit as st
import numpy as np
import model_chopin
from PIL import Image

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

st.title(':collision: Composer electronifire :collision:')

st.markdown('Please select an MC:')

if st.checkbox('Johann Sebastian Bach',st.image('bach.jpeg',width=100)):
     st.write('Good choice, but how about Chopin?')

if st.checkbox('Franz Schubert',st.image('schubert.jpeg',width=100)):
     st.write('Good choice, but how about Chopin?')

if st.checkbox('Frédéric Chopin',st.image('Freddy.jpeg',width=100)):
     if st.button('Electronify'):
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

         'Starting a long computation...'

           # Add a placeholder
         latest_iteration = st.empty()
         bar = st.progress(0)

         for i in range(100):
            # Update the progress bar with each iteration.
            latest_iteration.text(f'Iteration {i+1}')
            bar.progress(i + 1)
            time.sleep(0.5)

if st.checkbox('Claude Debussy',st.image('debussy.jpeg',width=100)):
     st.write('Good choice, but how about Chopin?')
