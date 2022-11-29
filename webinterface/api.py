from fastapi import FastAPI
import streamlit as st
import numpy as np
import model_chopin

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}

@app.get('/electronify')
def electronify():
    st.markdown('''#Composer electronifire

                Please select an MC''')

    if st.checkbox('Frédéric Chopin'):
        model_chopin
