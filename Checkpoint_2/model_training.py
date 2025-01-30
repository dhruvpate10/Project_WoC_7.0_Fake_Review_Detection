import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

data_path = '../Checkpoint_1/preprocessed_fake_reviews.csv'  # Update the path if necessary
data = pd.read_csv(data_path)

if 'processed_text' not in data.columns or 'label' not in data.columns:
    raise ValueError("Dataset must contain 'processed_text' and 'label' columns.")

data = data.dropna(subset=['processed_text'])

X = data['processed_text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


def train_and_evaluate_model(model, model_name):
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

    model_filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, model_filename)
    print(f"{model_name} saved as {model_filename}\n")


rf_model = RandomForestClassifier(random_state=42)
train_and_evaluate_model(rf_model, "Random Forest")

svm_model = SVC(kernel='linear', random_state=42)
train_and_evaluate_model(svm_model, "Support Vector Machine")

lr_model = LogisticRegression(random_state=42)
train_and_evaluate_model(lr_model, "Logistic Regression")

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("TF-IDF vectorizer saved as tfidf_vectorizer.pkl")
