import requests
import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read("./src/cred.cfg")


def fetch_yahoo_news(api_key, query):
    base_url = "https://newsapi.org/v2/everything"
    params = {"q": query, "apiKey": api_key}

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])
        df = pd.json_normalize(articles)
        # =============================================================================
        #         for article in articles:
        #             print(f"Title: {article['title']}")
        #             print(f"Source: {article['source']['name']}")
        #             print(f"Description: {article['description']}")
        #             print(f"URL: {article['url']}")
        #             print("-" * 50)
        #     else:
        #         print("Failed to fetch news:", response.text)
        # =============================================================================
        return df
    else:
        print("Failed to fetch news:", response.text)


if __name__ == "__main__":
    # Replace 'YOUR_API_KEY' with your actual Yahoo API key
    api_key = config["api"]["api_key_id"]
    query = (
        "Financial News"  # You can replace this query with your desired search query
    )
    df = fetch_yahoo_news(api_key, query)
    output_filename = "../Data/Prepared/CleanDatasets/financial_news_info.csv"
    df.to_csv(output_filename, index=False)
