import streamlit as st
from PIL import Image

def welcome():
    st.title("Welcome to Emotion code Analyzer")
    st.write("""Here you will find a tool to analyze the sentiment of financial news and tweets. 
             You can also provide a financial statement and get the sentiment of the statement.""")

    # Load image
    image = Image.open("App/Pages/testlogo.png") 

    # Display image centered
    st.image(image, use_column_width=True)

