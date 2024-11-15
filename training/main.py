# main.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_curve, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
import nltk
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Load NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

# Load data
data = pd.read_csv('extracted_metaphors.csv')
data = data.dropna(subset=['Word', 'Sentence']).reset_index(drop=True)

# Encode target variable
label_encoder = LabelEncoder()
data['Is_Metaphor_Label'] = label_encoder.fit_transform(data['Is_Metaphor'])

from pos_tagger import get_context_features, get_lemma_and_pos, get_pos_tags

# Extract features
print("Extracting context features...")
data['Context_Features'] = data.apply(get_context_features, axis=1)

print("Extracting target features...")
data['Target_Features'] = data['Word'].apply(get_lemma_and_pos)

print("Extracting context POS features...")
data['Context_POS'] = data.apply(get_pos_tags, axis=1)

# Vectorize features with limited number of features to speed up processing
print("Building feature matrix...")
vectorizer_word = TfidfVectorizer(max_features=5000)       # Limit features to 5000
vectorizer_context = TfidfVectorizer(max_features=5000)
vectorizer_pos = CountVectorizer(max_features=1000)

X_word = vectorizer_word.fit_transform(data['Target_Features'])
X_context = vectorizer_context.fit_transform(data['Context_Features'])
X_pos = vectorizer_pos.fit_transform(data['Context_POS'])

X = hstack([X_word, X_context, X_pos])
y = data['Is_Metaphor_Label']

# Split data into train and test sets, also keep data_test for generating .pos files
print("Splitting training and testing sets...")
X_train, X_test, y_train, y_test, data_train, data_test = train_test_split(
    X, y, data, test_size=0.2, stratify=y, random_state=42
)

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Define evaluation function
def evaluate_model(y_true, y_pred, target_names, title="Model Evaluation"):
    print(title)
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    print("-" * 50)

# Generate .pos files based on test set
def generate_predicted_labels(data_subset, predictions, filename='predicted_labels.pos'):
    with open(filename, 'w', encoding='utf-8') as f:
        for (idx, row), pred in zip(data_subset.iterrows(), predictions):
            word = str(row['Word']).strip()
            # Handle empty or special characters
            if not word:
                word = 'UNK'
            lemma_pos = get_lemma_and_pos(word)
            # Parse lemma and pos
            if '_' in lemma_pos:
                parts = lemma_pos.split('_', 1)
                lemma = parts[0] if len(parts) > 0 else 'UNK'
                pos = parts[1] if len(parts) > 1 else 'UNK'
            else:
                lemma = lemma_pos
                pos = 'UNK'
            # Handle empty fields
            lemma = lemma if lemma else 'UNK'
            pos = pos if pos else 'UNK'
            is_metaphor = pred
            metaphor_label = 'True' if is_metaphor == 1 else 'False'

            # Ensure four fields per line
            f.write(f"{word}\t{lemma}\t{pos}\t{metaphor_label}\n")
    print(f"Predicted labels file generated: {filename}")

def generate_annotation_gold(data_subset, filename='annotation_gold.pos'):
    with open(filename, 'w', encoding='utf-8') as f:
        for idx, row in data_subset.iterrows():
            word = str(row['Word']).strip()
            # Handle empty or special characters
            if not word:
                word = 'UNK'
            lemma_pos = get_lemma_and_pos(word)
            # Parse lemma and pos
            if '_' in lemma_pos:
                parts = lemma_pos.split('_', 1)
                lemma = parts[0] if len(parts) > 0 else 'UNK'
                pos = parts[1] if len(parts) > 1 else 'UNK'
            else:
                lemma = lemma_pos
                pos = 'UNK'
            # Handle empty fields
            lemma = lemma if lemma else 'UNK'
            pos = pos if pos else 'UNK'
            is_metaphor = row['Is_Metaphor_Label']
            metaphor_label = 'True' if is_metaphor == 1 else 'False'

            # Ensure four fields per line
            f.write(f"{word}\t{lemma}\t{pos}\t{metaphor_label}\n")
    print(f"Gold standard labels file generated: {filename}")

# Train initial Logistic Regression model
print("Training initial Logistic Regression model...")
clf_lr = LogisticRegression(class_weight=class_weights_dict, max_iter=1000, random_state=42)
clf_lr.fit(X_train, y_train)

# Predict and evaluate
print("Predicting...")
y_pred_lr = clf_lr.predict(X_test)
evaluate_model(y_test, y_pred_lr, ['Non-metaphor', 'Metaphor'], "Logistic Regression Report")

# Adjust threshold
print("Adjusting classification threshold...")
y_probs_lr = clf_lr.predict_proba(X_test)[:, 1]

# Compute precision-recall curve
precision_curve, recall_curve, thresholds_curve = precision_recall_curve(y_test, y_probs_lr)

# Plot and save the curve
plt.figure(figsize=(8, 6))
plt.plot(thresholds_curve, precision_curve[:-1], label='Precision')
plt.plot(thresholds_curve, recall_curve[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold')
plt.legend()
plt.savefig('precision_recall_vs_threshold.png')
plt.close()

# Find the best threshold based on F1 score
best_f1, best_threshold = 0, 0.5
for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_adjusted = (y_probs_lr >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_adjusted)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold}, Corresponding F1-score: {best_f1:.4f}")

# Predict with the best threshold
y_pred_adjusted_lr = (y_probs_lr >= best_threshold).astype(int)
evaluate_model(y_test, y_pred_adjusted_lr, ['Non-metaphor', 'Metaphor'], f"Logistic Regression (Threshold {best_threshold}) Report")

# Train with SMOTE
print("Training with SMOTE...")
smote = SMOTE(random_state=42)

# Use Pipeline to integrate SMOTE and Logistic Regression
pipeline_smote_lr = ImbPipeline([
    ('smote', smote),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
pipeline_smote_lr.fit(X_train, y_train)

# Predict and evaluate
y_pred_smote_lr = pipeline_smote_lr.predict(X_test)
evaluate_model(y_test, y_pred_smote_lr, ['Non-metaphor', 'Metaphor'], "SMOTE + Logistic Regression Report")

# Generate .pos files based on the test set
print("Generating .pos files based on the test set...")
generate_predicted_labels(data_test, y_pred_smote_lr, filename='predicted_labels.pos')
generate_annotation_gold(data_test, filename='annotation_gold.pos')

# Save predictions for the test set
print("Saving predictions to CSV...")
data_test['Is_Metaphor_Pred'] = y_pred_smote_lr
data_test.to_csv('predictions_test.csv', index=False)

print("Process completed.")
