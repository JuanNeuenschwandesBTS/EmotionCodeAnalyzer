# Financial Sentiment Analysis Project Documentation

## Raw Data Overview

This documentation provides a comprehensive overview of the raw data used in the financial sentiment analysis and category tagging project. It outlines the source, content, and intended use of each dataset, highlighting their roles in training and evaluating models for both sentiment analysis and topic classification.

### 1. Financial Phrasebank Dataset (`fin_phrase_bank.csv`)

- **Description**: Contains sentences extracted from financial news articles, along with their associated sentiment labels, aimed at sentiment analysis within the financial domain.
- **Structure**:
  - Entries: 4,846
  - Columns: 2 (sentence, label)
- **Content**:
  - `sentence`: Text of the sentence extracted from financial news.
  - `label`: Sentiment label assigned to each sentence (e.g., positive, negative, neutral), represented as integers.
- **Application**: Used for training and evaluating models for sentiment analysis, specifically assessing the sentiment of financial texts.

### 2. Twitter Financial News Topic Dataset for Training (`topic_train.csv`)

- **Description**: Comprises financial news texts extracted from Twitter, labeled with topics relevant to the financial industry, designated for training topic classification models.
- **Structure**:
  - Entries: 16,990
  - Columns: 2 (text, label)
- **Content**:
  - `text`: The text of the tweet related to financial news.
  - `label`: Topic label assigned to each tweet, indicating the financial topic it pertains to, represented as integers.
- **Application**: Utilized to train NLP classifiers capable of assigning financial categories to texts, aiding in the categorization of finance-related tweets by topic.

### 3. Twitter Financial News Topic Dataset for Validation (`topic_valid.csv`)

- **Description**: Similar to the training dataset, this validation set contains tweets related to financial news, labeled with financial topics, and is used for model validation.
- **Structure**:
  - Entries: 4,117
  - Columns: 2 (text, label)
- **Content**: Same as the training dataset, with text representing the tweet text and label the associated financial topic.
- **Application**: Provides a separate dataset for validating the performance of trained topic classification models, ensuring they accurately classify new, unseen financial texts.

---

## Step 1: Preprocessing

### Data Overview

This dataset, `FinancialLogitReg_tf_idf.csv`, is part of a financial sentiment analysis project. It consists of 1,454 entries, each representing a piece of financial text derived from various sources. The dataset is structured into three columns:

- `text`: Contains the cleaned and pre-processed financial text.
- `true_label`: The actual sentiment/category label assigned to the text.
- `pred_label`: The predicted sentiment/category label from the logistic regression model using TF-IDF vectorization.

### Data Cleaning and Pre-processing

The dataset underwent several cleaning and pre-processing steps before being used for analysis, including normalization, punctuation removal, tokenization, stop word removal, and lemmatization.

### TF-IDF Vectorization

TF-IDF vectorization was applied to the pre-processed text to extract features for the logistic regression model, highlighting important words and converting text data into a numerical format suitable for analysis.

---

## Model Training and Evaluation

### Baseline Model: TF-IDF and Logistic Regression

This model used TF-IDF for feature extraction and Logistic Regression for classification. It demonstrated a reasonable performance with an accuracy of around 0.74, indicating the effectiveness of this approach in classifying financial sentiments.

### Word2Vec and Logistic Regression

Another baseline model employed Word2Vec for feature extraction and Logistic Regression for classification, aiming to capture semantic meaning more effectively. It addressed the convergence issue by potentially increasing the number of iterations or adjusting the model's regularization strength.

### SVC Model: Sentiment Analysis

The SVC model with a linear kernel was trained on the Financial Phrasebank data, achieving precision scores of approximately 0.786 for neutral, 0.694 for negative, and 0.760 for positive sentiments.

### BERT Fine-Tuning Model

The model was initialized with the `bert-base-uncased` pre-trained model and fine-tuned on financial text data. It demonstrated the ability to classify sentiments with a validation accuracy of around 0.66, highlighting its potential for improving sentiment analysis accuracy in financial texts.

---

## Observations and Conclusions

Throughout the project, various models were evaluated, including logistic regression with TF-IDF and Word2Vec features, an SVM classifier, and a fine-tuning of a BERT model. The evaluation process highlighted the importance of feature extraction techniques and model choice in financial sentiment analysis, with each model offering insights into the challenges and opportunities in classifying financial sentiments accurately.

