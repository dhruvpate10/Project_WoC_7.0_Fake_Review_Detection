import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

data_path = 'fakeReviewData.csv'
data = pd.read_csv(data_path)

print("Dataset Shape:", data.shape)
print("Dataset Columns:\n", data.columns)
print("First 5 Rows:\n", data.head())

data.dropna(inplace=True)

data.drop_duplicates(inplace=True)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


data['cleaned_text'] = data['text_'].apply(preprocess_text)

data['tokens'] = data['cleaned_text'].apply(word_tokenize)

stop_words = set(stopwords.words('english'))
data['tokens'] = data['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

stemmer = PorterStemmer()
data['tokens'] = data['tokens'].apply(lambda x: [stemmer.stem(word) for word in x])

data['processed_text'] = data['tokens'].apply(lambda x: ' '.join(x))

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['processed_text'])

vectorized_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

preprocessed_data_path = 'preprocessed_fake_reviews.csv'
data.to_csv(preprocessed_data_path, index=False)

print(f"Preprocessed data saved to {preprocessed_data_path}")
