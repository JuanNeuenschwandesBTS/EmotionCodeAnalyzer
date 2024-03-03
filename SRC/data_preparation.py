"""
# Data Pre-processing

----------

Download and inspect the data from the various sources:

1. Financial Phrasebank https://huggingface.co/datasets/financial_phrasebank. Humanly annotated

2. Financial tweets topics dataset: https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic/viewer/default/train?p=169. Humanly annotated

Think of any pre-processing functions (
    Converting the text to lowercase,
    removing punctuation,
    tokenizing the text,
    removing stop words and empty strings,
    lemmatizing tokens.
) that you might need to apply for downstream tasks. As always, pick a framework for data analysis and data exploration.

"""

import pandas as pd
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def clean_text(text):
    # remove URLS
    # text = re.sub(r'http\S+', ' ', text)
    # remove url.com
    # text = re.sub(r'\w+\.com', ' ', text)
    # replace % with percent
    # text = re.sub(r'%', ' percent', text)
    # remove punctuation at the end of words
    text = re.sub(r"[.,!?]", " ", text)
    # remove special characters except fo decimal points
    text = re.sub(r"[^a-zA-Z\s.]", " ", text)
    # remove extra spaces
    text = re.sub(r"\s+", " ", text)
    # remove leading and trailing spaces
    text = re.sub(r"^\s+|\s+?$", " ", text.lower())

    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"RT[\s]+", "", text)
    text = re.sub(r"https?:\/\/\S+", "", text)
    return text


def remove_stopwords(text):
    words = [word for word in text if word not in stopwords.words("english")]
    tex = " ".join(words)
    return tex


def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split(" ")]
    tex = " ".join(words)
    return tex


def preprocess_text(text):
    text = text.lower()
    text = clean_text(text)
    text = word_tokenize(text)
    text = remove_stopwords(text)
    text = lemmatize(text)

    return text


def writeDataToDisk(output_filename, df):
    # TO-DO
    df.to_csv(output_filename, index=False)


topics = {
    0: "Analyst Update",
    1: "Fed | Central Banks",
    2: "Company | Product News",
    3: "Treasuries | Corporate Debt",
    4: "Dividend",
    5: "Earnings",
    6: "Energy | Oil",
    7: "Financials",
    8: "Currencies",
    9: "General News | Opinion",
    10: "Gold | Metals | Materials",
    11: "IPO",
    12: "Legal | Regulation",
    13: "M&A | Investments",
    14: "Macro",
    15: "Markets",
    16: "Politics",
    17: "Personnel Change",
    18: "Stock Commentary",
    19: "Stock Movement",
}

sentiment = {0: "Negative", 1: "Neutral", 2: "Positive"}


if __name__ == "__main__":
    dataset = load_dataset("financial_phrasebank", "sentences_50agree")
    df_fin_phrase = pd.DataFrame(dataset["train"])
    df_tweet_topic_train = pd.read_csv("../Data/Raw/topic_train.csv")
    df_tweet_topic_valid = pd.read_csv("../Data/Raw/topic_valid.csv")
    df_fin_phrase = pd.read_csv("../Data/Raw/fin_phrase_bank.csv")
    df_tweet_topic_train["clean_text"] = df_tweet_topic_train["text"].apply(
        preprocess_text
    )
    df_tweet_topic_valid["clean_text"] = df_tweet_topic_valid["text"].apply(
        preprocess_text
    )

    df_tweet_topic_valid["topic"] = df_tweet_topic_valid["label"].apply(
        lambda x: topics[x]
    )
    df_tweet_topic_train["topic"] = df_tweet_topic_train["label"].apply(
        lambda x: topics[x]
    )
    df_fin_phrase["sentiment"] = df_fin_phrase["label"].apply(lambda x: sentiment[x])

    writeDataToDisk(
        "../Data/Prepared/CleanDatasets/Tweet_valid_clean.csv", df_tweet_topic_valid
    )
    writeDataToDisk(
        "../Data/Prepared/CleanDatasets/Tweet_train_clean.csv", df_tweet_topic_train
    )
    writeDataToDisk(
        "../Data/Prepared/CleanDatasets/fin_phrase_bank_clean.csv", df_fin_phrase
    )
