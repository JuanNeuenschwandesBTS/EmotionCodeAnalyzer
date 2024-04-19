import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from financial_news_api import NewApis


def page_sentiment_classifier():
    financial_news = NewApis()

    # Load the tokenizer and the model

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)


    @st.cache_resource()
    def load_model():
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        return model


    # Load the model
    model = load_model()
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Define the Streamlit app
    st.title("Financial Sentiment App developed by Juan, Ciaran and John")


    def get_news_api():
        df = financial_news.fetch_yahoo_news()
        df["label"] = df["description"].apply(lambda x: pipe(x)[0].get("label"))
        df["score"] = df["description"].apply(lambda x: pipe(x)[0].get("score"))
        st.dataframe(df)


    selection = st.radio(
        "Select option: ",
        [
            "Get News of the Day Sentiment and the sentiment",
            "Provide a Financial Statement for Sentiment Analysis",
        ],
    )

    if selection == "Provide a Financial Statement":
        tweet = st.text_area("Enter a tweet:", "")
        button_clicked = st.button("Submit")
        if button_clicked and tweet is not None:
            outputs = pipe(tweet)

            # Display the prediction
            st.write(f"The Sentiment is: {outputs[0].get('label')}")
            st.write(f"The Confidence is: {outputs[0].get('score')}")
    if selection == "Get News of the Day Sentiment":
        with st.spinner("Loading..."):
            get_news_api()
        # button_clicked = st.button("Submit")
        # if button_clicked:
        # get_news_api()