# Project WoC 7.0: Fake Review Detection

This repository contains the code and data for the first checkpoint of the Fake Review Detection project. The objective of this checkpoint is to preprocess the raw dataset to prepare it for machine learning tasks.

## Preprocessing Steps

### 1. Load Dataset
- The dataset (`fakeReviewData.csv`) is loaded and explored to understand its structure and contents.

### 2. Data Cleaning
- **Handle Missing Values**: Rows with missing values are removed.
- **Remove Duplicates**: Duplicate rows are dropped to ensure data quality.

### 3. Text Normalization
- Convert all text to lowercase.
- Remove punctuation, special characters, and numbers.

### 4. Tokenization
- Split text into individual words (tokens) for analysis.

### 5. Stopword Removal
- Common words that do not contribute to meaning (e.g., "and," "the") are removed using the NLTK library.

### 6. Stemming
- Words are reduced to their root form (e.g., "running" becomes "run") using the Porter Stemmer.

### 7. Vectorization
- The preprocessed text is converted into numerical format using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

### 8. Save Processed Data
- The cleaned and processed dataset is saved as `preprocessed_fake_reviews.csv` for further use.

## Repository Structure
- `fake_review_preprocessing.py`: Python script containing the preprocessing code.
- `preprocessed_fake_reviews.csv`: Preprocessed dataset.
- `README.md`: Documentation for the repository.

## Requirements
To run the preprocessing code, the following Python libraries are required:
- pandas
- numpy
- re
- nltk
- scikit-learn

## How to Run
1. Place the raw dataset (`fakeReviewData.csv`) in the project directory.
2. Execute the `fake_review_preprocessing.py` script.
3. The processed dataset will be saved as `preprocessed_fake_reviews.csv` in the same directory.

## Acknowledgments
This project is part of the WoC 7.0 initiative and is aimed at developing a machine learning model for detecting fake reviews.

