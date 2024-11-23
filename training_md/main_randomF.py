import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import string
import csv
import os

# Download necessary resources
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# Function to load data
def load_data(file_path):
    annotated_data = []
    print(f"Loading data from: {file_path}")
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 5:
                    annotated_data.append(tuple(parts))  # (word, lemma, pos, sentence, label)
        print(f"Loaded {len(annotated_data)} samples.")
    except Exception as e:
        print(f"Error loading data: {e}")
    return annotated_data

# Function to extract features
def extract_features(index, tokens, word_frequencies, sentiment_analyzer):
    word, lemma, pos, sentence, _ = tokens[index]
    word_clean = word.lower().strip(string.punctuation)
    
    features = {
        'word': word_clean,
        'lemma': lemma.lower(),
        'pos': pos,
        'is_capitalized': word[0].isupper(),
        'is_stopword': word_clean in stopwords.words('english'),
        'word_frequency': word_frequencies.get(word_clean, 0),
    }
    
    # Add context features
    if index > 0:
        features['prev_word'] = tokens[index - 1][0].lower()
        features['prev_pos'] = tokens[index - 1][2]
    else:
        features['prev_word'] = 'START'
        features['prev_pos'] = 'START_POS'
    
    if index < len(tokens) - 1:
        features['next_word'] = tokens[index + 1][0].lower()
        features['next_pos'] = tokens[index + 1][2]
    else:
        features['next_word'] = 'END'
        features['next_pos'] = 'END_POS'
    
    return features

# Prepare data for training
def prepare_data(annotated_data, word_frequencies, sentiment_analyzer):
    prepared_data = []
    for i in range(len(annotated_data)):
        features = extract_features(i, annotated_data, word_frequencies, sentiment_analyzer)
        label = annotated_data[i][4]  # The label is now at index 4
        prepared_data.append((features, label))
    return prepared_data

# Get word frequencies
def get_word_frequencies(data):
    word_freq = Counter(word.lower() for word, _, _, _, _ in data)
    return word_freq

# Train and evaluate Random Forest
def train_random_forest(X, y):
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, class_weight='balanced')
    model.fit(X, y)
    return model

# Optimize threshold
def evaluate_with_threshold(model, X, y_true):
    print("Optimizing threshold...")
    y_probs = model.predict_proba(X)[:, 1]  # Get probabilities for the "True" class
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Precision-Recall vs Threshold")
    plt.legend()
    plt.show()

    # Find the threshold with the highest precision
    best_threshold_index = precisions.argmax()
    best_threshold = thresholds[best_threshold_index]
    print(f"Best Threshold: {best_threshold}, Precision: {precisions[best_threshold_index]}, Recall: {recalls[best_threshold_index]}")
    return best_threshold

# Run the full pipeline
def run_pipeline(train_file, dev_file, output_file):
    # Load training data
    train_data = load_data(train_file)
    if not train_data:
        print("No training data available.")
        return

    # Feature preparation
    word_frequencies = get_word_frequencies(train_data)
    sentiment_analyzer = SentimentIntensityAnalyzer()
    prepared_data = prepare_data(train_data, word_frequencies, sentiment_analyzer)

    # Convert features to numerical format using DictVectorizer
    print("Converting features to vectors...")
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform([features for features, _ in prepared_data])
    y = [1 if label == 'True' else 0 for _, label in prepared_data]  # Convert labels to binary

    # Train-test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest
    model = train_random_forest(X_train, y_train)

    # Optimize and apply threshold
    threshold = evaluate_with_threshold(model, X_test, y_test)

    # Evaluate model
    print("Evaluating model with optimized threshold...")
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    print(classification_report(y_test, y_pred))
    print(f"Macro F1 Score: {f1_score(y_test, y_pred, average='macro')}")

    # Predictions on dev file
    if os.path.exists(dev_file):
        predict_and_save(model, dev_file, output_file, sentiment_analyzer, word_frequencies, vectorizer, threshold)
    else:
        print(f"Dev file '{dev_file}' not found.")

# Prediction and save function
def predict_and_save(model, input_file, output_file, sentiment_analyzer, word_frequencies, vectorizer, threshold):
    print(f"Predicting on {input_file} and saving to {output_file}")
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
            reader = csv.reader(infile, delimiter='\t')
            writer = csv.writer(outfile, delimiter='\t')

            for row in reader:
                if len(row) < 3:  # Skip malformed rows
                    writer.writerow(row)
                    continue

                word, lemma, pos = row[:3]
                token_data = [(word, lemma, pos, None, None)]  # Modify to include all columns
                features = extract_features(0, token_data, word_frequencies, sentiment_analyzer)
                feature_vector = vectorizer.transform([features])
                prediction_prob = model.predict_proba(feature_vector)[0][1]
                prediction = 'True' if prediction_prob >= threshold else 'False'

                writer.writerow([word, lemma, pos, 'sentence', prediction])  # Write sentence as well
    except Exception as e:
        print(f"Error during prediction: {e}")

# Main execution
if __name__ == "__main__":
    train_file = 'train.pos'
    dev_file = 'dev_to_annotate.pos'
    output_file = 'dev_annotated.pos'

    if os.path.exists(train_file):
        run_pipeline(train_file, dev_file, output_file)
    else:
        print(f"Error: {train_file} does not exist.")
