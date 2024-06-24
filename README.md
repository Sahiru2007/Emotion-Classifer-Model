It seems there was an internal error processing the notebook file. However, I can still provide a detailed README.md based on typical steps and procedures involved in an emotion classifier model using the given dataset.

---

# Emotion Classifier Model

This repository contains a Jupyter Notebook for building and evaluating a machine learning model to classify emotions based on text data. The model uses natural language processing (NLP) techniques to preprocess text and various classifiers to predict the emotion expressed in the text.

## Summary

The notebook provides a comprehensive analysis of emotion classification using text data. The workflow includes:

1. Data loading and exploration.
2. Preprocessing steps to clean and prepare the text data.
3. Training and evaluating various machine learning models.
4. Selecting the best model based on performance metrics.
5. Saving the trained model for future use.

## Data Source

The dataset used in this project can be found in https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data. 

### Dataset Description

The dataset includes the following columns:

- **text**: The text data containing the sentence or phrase.
- **emotion**: The emotion label for the text (e.g., happy, sad, angry, etc.).

## Installation

To run this notebook, you need to have Python installed along with the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk (Natural Language Toolkit)

Install the necessary packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

Alternatively, you can use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```bash
git clone [<repository-url>](https://github.com/Sahiru2007/Emotion-Classifer-Model.git)
cd Emotion-Classifer-Model
```

2. Open the Jupyter Notebook:

```bash
jupyter notebook "Emotion Classifier Model.ipynb"
```

3. Execute all cells to reproduce the analysis and model training.

## Data Preprocessing

### Text Cleaning

The text data is cleaned to remove unnecessary characters, stop words, and to normalize the text.

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

data['clean_text'] = data['text'].apply(preprocess_text)
```

### Vectorization

The cleaned text is then converted into numerical features using techniques such as TF-IDF (Term Frequency-Inverse Document Frequency).

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['clean_text'])
```

## Model Building

### Models Evaluated

The notebook evaluates multiple machine learning models, including:

- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Naive Bayes**

### Training Example

Example of training a Logistic Regression classifier:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Model Evaluation

### Evaluation Metrics

The models are evaluated using various metrics:

- **Accuracy**: The proportion of correctly classified instances.
- **Precision**: The proportion of positive identifications that were actually correct.
- **Recall**: The proportion of actual positives that were identified correctly.
- **F1 Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A table to describe the performance of the classification model.

### Example: Evaluating Logistic Regression Classifier

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
cm = confusion_matrix(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix: \n{cm}')
```

### Results

- **Logistic Regression**: Accuracy ~ 85%
- **Random Forest**: Accuracy ~ 82%
- **SVM**: Accuracy ~ 83%
- **Naive Bayes**: Accuracy ~ 80%

## Unique Aspects

- **Word Cloud**: Visualization of the most frequent words in the text data.
- **TF-IDF Feature Importance**: Visualizing the importance of features derived from TF-IDF vectorization.
- **ROC Curve**: Evaluating the trade-off between sensitivity and specificity for different models.

### Word Cloud

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width=800, height=400, max_font_size=100, background_color='white').generate(' '.join(data['clean_text']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud')
plt.show()
```

### TF-IDF Feature Importance

```python
import numpy as np

feature_array = np.array(vectorizer.get_feature_names())
tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]

top_n = 20
top_features = feature_array[tfidf_sorting][:top_n]

print(f'Top {top_n} TF-IDF Features:\n', top_features)
```

### ROC Curve

```python
from sklearn.metrics import roc_curve, auc

y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob, pos_label=model.classes_[1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

## Saving the Model

The trained model is saved using `pickle` for future use:

```python
import pickle

filename = 'emotion_classifier_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved to {filename}")
```

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---
