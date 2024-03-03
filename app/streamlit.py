import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd
import glob

# Load the tokenizer and the model

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)


@st.cache(allow_output_mutation=True)
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return pipe


# Load the model
model = load_model()

# Define the Streamlit app
st.title("New Sentiment App")


tweet = st.text_area("Enter a tweet:", "")

if st.button("Predict"):
    # Preprocess the tweet
    inputs = tokenizer(tweet, return_tensors="pt")

    # Get model's prediction
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs)

    # Display the prediction
    if prediction == 0:
        st.write("The tweet has a positive sentiment.")
    else:
        st.write("The tweet has a negative sentiment.")
