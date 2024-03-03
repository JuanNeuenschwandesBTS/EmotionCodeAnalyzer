import requests
import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read("App/cred.cfg")


class NewApis:
    def __init__(self):
        self.api_key = config["api"]["api_key_id"]

    def fetch_yahoo_news(self):
        query = "Financial News"  # You can replace this query with your desired search query
        base_url = "https://newsapi.org/v2/everything"
        params = {"q": query, "apiKey": self.api_key}

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            df = pd.json_normalize(articles)
            return df
        else:
            print("Failed to fetch news:", response.text)


# =============================================================================
#
# if __name__ == "__main__":
#     # Replace 'YOUR_API_KEY' with your actual Yahoo API key
#     df = fetch_yahoo_news()
#     output_filename = "../Data/Prepared/CleanDatasets/financial_news_info.csv"
#     df.to_csv(output_filename, index=False)
# =============================================================================
