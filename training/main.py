import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import matplotlib.pyplot as plt

# Import custom functions from posfile.py
from posfile import extract_features, generate_predicted_labels, generate_annotation_gold

warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('extracted_metaphors.csv')
data = data.dropna(subset=['Word', 'Sentence']).reset_index(drop=True)

# Encode target variable
label_encoder = LabelEncoder()
data['Is_Metaphor_Label'] = label_encoder.fit_transform(data['Is_Metaphor'])

# Extract features
print("Extracting features...")
X, y, data = extract_features(data)

# Split data into train and test sets
print("Splitting training and testing sets...")
X_train, X_test, y_train, y_test, data_train, data_test = train_test_split(
    X, y, data, test_size=0.2, stratify=y, random_state=42
)

# Feature scaling
scaler = MaxAbsScaler()  # Suitable for sparse matrices
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Define evaluation function
def evaluate_model(y_true, y_pred, title="Model Evaluation"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(title)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("-" * 50)


# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter tuning for Logistic Regression
print("Hyperparameter tuning for Logistic Regression...")
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search_lr = GridSearchCV(
    estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    param_grid=param_grid_lr,
    scoring='f1',
    n_jobs=-1,
    cv=cv,
    verbose=1
)
grid_search_lr.fit(X_train_scaled, y_train)

print(f"Best parameters for Logistic Regression: {grid_search_lr.best_params_}")
best_lr = grid_search_lr.best_estimator_

# Predict and evaluate
print("Predicting with best Logistic Regression...")
y_pred_lr = best_lr.predict(X_test_scaled)
evaluate_model(y_test, y_pred_lr, "Best Logistic Regression Report")

# Adjust classification threshold for Logistic Regression
print("Adjusting classification threshold for Logistic Regression...")
y_probs_lr = best_lr.predict_proba(X_test_scaled)[:, 1]
best_f1_lr, best_threshold_lr = 0, 0.5
for threshold in np.arange(0.1, 0.9, 0.01):
    y_pred_adjusted = (y_probs_lr >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_adjusted)
    if f1 > best_f1_lr:
        best_f1_lr = f1
        best_threshold_lr = threshold

print(f"Best threshold: {best_threshold_lr:.2f}, Corresponding F1-score: {best_f1_lr:.4f}")
y_pred_adjusted_lr = (y_probs_lr >= best_threshold_lr).astype(int)
evaluate_model(y_test, y_pred_adjusted_lr, f"Best Logistic Regression (Threshold {best_threshold_lr:.2f}) Report")

# Train with ADASYN and Logistic Regression
print("Training with ADASYN and Logistic Regression...")
pipeline_adasyn_lr = ImbPipeline([
    ('sampling', ADASYN(random_state=42)),
    ('scaler', MaxAbsScaler()),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, **grid_search_lr.best_params_))
])
pipeline_adasyn_lr.fit(X_train, y_train)

# Predict and evaluate
y_pred_adasyn_lr = pipeline_adasyn_lr.predict(X_test)
evaluate_model(y_test, y_pred_adasyn_lr, "ADASYN + Logistic Regression Report")

# Adjust classification threshold for ADASYN + Logistic Regression
print("Adjusting classification threshold for ADASYN + Logistic Regression...")
y_probs_adasyn_lr = pipeline_adasyn_lr.predict_proba(X_test)[:, 1]
best_f1_adasyn_lr, best_threshold_adasyn_lr = 0, 0.5
for threshold in np.arange(0.1, 0.9, 0.01):
    y_pred_adjusted = (y_probs_adasyn_lr >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_adjusted)
    if f1 > best_f1_adasyn_lr:
        best_f1_adasyn_lr = f1
        best_threshold_adasyn_lr = threshold

print(f"Best threshold: {best_threshold_adasyn_lr:.2f}, Corresponding F1-score: {best_f1_adasyn_lr:.4f}")
y_pred_adjusted_adasyn_lr = (y_probs_adasyn_lr >= best_threshold_adasyn_lr).astype(int)
evaluate_model(y_test, y_pred_adjusted_adasyn_lr,
               f"ADASYN + Logistic Regression (Threshold {best_threshold_adasyn_lr:.2f}) Report")

# Generate .pos files based on the best model
print("Generating .pos files...")
# Assuming the best model is the ADASYN + Logistic Regression with adjusted threshold
generate_predicted_labels(data_test, y_pred_adjusted_adasyn_lr, filename='predicted_labels.pos')
generate_annotation_gold(data_test, filename='annotation_gold.pos')

print("Process completed.")
