# Model Training for Fake Review Detection

This README provides an overview of the model training process for Fake Review Detection, including dataset preparation, model selection, evaluation, and final recommendations.

## **1. Dataset Preparation**
- The dataset used for model training is `preprocessed_fake_reviews.csv`.
- The dataset was split into **80% training** and **20% testing** to evaluate model performance.
- Text data was converted into numerical format using **TF-IDF vectorization**.

## **2. Model Selection & Performance**
Three machine learning models were trained and evaluated:

| Model                  | Accuracy | Precision | Recall  | F1 Score |
|------------------------|----------|------------|---------|-----------|
| Random Forest         | 84.74%   | 85.01%     | 84.74%  | 84.70%    |
| Support Vector Machine | 87.22%   | 87.22%     | 87.22%  | 87.22%    |
| Logistic Regression   | 86.55%   | 86.57%     | 86.55%  | 86.55%    |

## **3. Best Model Recommendation**
Based on the evaluation metrics:
- **Support Vector Machine (SVM) performed the best** with the highest accuracy (87.22%) and the best-balanced Precision, Recall, and F1 Score.
- **SVM is recommended for Fake Review Detection** due to its superior ability to differentiate between real and fake reviews.

## **4. Model Serialization**
The trained models were saved for future use:
- `random_forest_model.pkl`
- `support_vector_machine_model.pkl`
- `logistic_regression_model.pkl`
- `tfidf_vectorizer.pkl` (for text transformation)

## **5. How to Use the Models**
To load and use a trained model:
```python
import joblib

# Load the SVM model
svm_model = joblib.load("support_vector_machine_model.pkl")

# Load the vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Example Prediction
sample_text = ["This product is amazing! Highly recommend it."]
sample_vectorized = vectorizer.transform(sample_text)
prediction = svm_model.predict(sample_vectorized)
print("Predicted Label:", prediction)
```

## **6. Next Steps**
- Further tuning of hyperparameters could improve performance.
- More advanced models such as deep learning approaches could be explored.
- Implementing real-time fake review detection using this model.

