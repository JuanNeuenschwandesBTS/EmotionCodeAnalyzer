import streamlit as st
from Pages.FinancialSentimentClassifier import page_sentiment_classifier 
from Pages.welcome import welcome
import time

def main():
    
    
    selection = 'Welcome'
    st.sidebar.title('Menu')
    selection = st.sidebar.radio('Go to', ['Welcome','Sentiment Classifier'])
    if selection == 'Welcome':  
        welcome()
    elif selection == 'Sentiment Classifier':
        with st.spinner('Loading...'):
            time.sleep(5)
        page_sentiment_classifier()
    # elif selection == 'Page 2':
    #     page2.page2()
    # Footer
    

if __name__ == '__main__':
    main()