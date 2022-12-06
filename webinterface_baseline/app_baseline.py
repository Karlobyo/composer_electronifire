
import streamlit as st
import base64

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False
def callback():
    st.session_state.button_clicked = True

if "persisted_variable" not in st.session_state:
    st.session_state.persisted_variable = 0



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

def header(url):
    st.markdown(f'<p style="color:#FFFFFF;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)



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

st.title(':collision: Composer Electronifier :collision:')

st.markdown('Please select a composer:')


if st.checkbox('Johann Sebastian Bach',value=False,args=st.image('bach.jpeg',width=100)):
     st.write('Good choice, but how about Chopin?')

if st.checkbox('Franz Schubert',value=False,args=st.image('schubert.jpeg',width=100)):
     st.write('Good choice, but how about Chopin?')

if st.checkbox('Frédéric Chopin',value=False,args=st.image('Freddy.jpeg',width=100)):
    if st.button(label='Generate a new melody'):
        st.session_state.persisted_variable += 1

    if st.session_state.persisted_variable == 1:
        st.write('🎶 Listen to your new tune 🎶')
        st.audio(data='Chopin_J_base.wav',format='audi/wav')

        beat = st.selectbox(label='Choose a 🥁 beat 🥁 to electroni-🔥 your tune.',options=['Triphop', 'Techno',], on_change = None,key=1)

        electronifire = st.button('Electroni-🔥!')

        if electronifire:
            add_bg_from_local('discoball2.jpg')
            if beat == 'Triphop':
                st.audio(data='compelec1_triphop_quiet_beat1.wav',format='audio/wav')
            if beat == 'Techno':
                st.audio(data='compelec1_technobeat.wav',format='audio/wav')


    # st.write('Listen to medlodies in the style of Chopin. You can choose between a basic AI-generated melody or an advanced model-generated melody')
    # melody = st.selectbox(label='Choose an AI-generated melody',options=['Basic melody #1', 'Basic melody #2', 'Advanced melody #1'])
    # if melody == 'Basic melody #1':
    #     st.audio(data='Mel1.mid.wav',format='audi/wav')
    #     st.image(image='music_sheet_#1.png',width=500)

    # if melody == 'Basic melody #2':
    #     st.audio(data='Mel2.mid.wav',format='audi/wav')
    #     st.image(image='music_sheet_#2.png',width=500)

    # if melody == 'Advanced melody #1':
    #     st.audio(data='Chopin_J_base.wav',format='audi/wav')

    # beat = st.selectbox(label='Choose a beat to electroni-🔥 your tune.',options=['Beat #1',], on_change = None,key=1)
    #      #on_change will call the second API with the argument that is chosen by user

    # electronifire = st.button('Electroni-🔥!')

    # if electronifire:

    #     if beat ==
    #     st.audio(data='beat1compelec_YES.wav',format='audio/wav')




if st.checkbox('Claude Debussy',value=False,args=st.image('debussy.jpeg',width=100)):
     st.write('Good choice, but how about Chopin?')
