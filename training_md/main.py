import nltk
from nltk.classify import MaxentClassifier
import csv
import os
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

# Ensure you have these NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

def load_data(file_path):
    annotated_data = []
    print(f"Attempting to load data from: {file_path}")
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines between sentences
                
                parts = line.split('\t')
                if len(parts) == 5:  # Adjusted to account for the extra sentence column
                    word, lemma, pos, sentence, label = parts
                    annotated_data.append((word, lemma, pos, sentence, label))
                else:
                    print(f"Warning: Line {line_num} does not have 5 parts: {line}")
        
        print(f"Loaded {len(annotated_data)} samples from {file_path}")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
    
    return annotated_data

def extract_features(index, tokens, word_frequencies, sentiment_analyzer):
    word, lemma, pos, sentence, _ = tokens[index]  # Account for the sentence column
    features = {
        'word': word,
        'lemma': lemma,
        'pos': pos,
        'sentence': sentence,  # Adding the sentence as a feature
    }
    
    if index > 0:
        features['prev_word'] = tokens[index - 1][0]
        features['prev_lemma'] = tokens[index - 1][1]
        features['prev_pos'] = tokens[index - 1][2]
    else:
        features['prev_word'] = 'START'
    
    if index < len(tokens) - 1:
        features['next_word'] = tokens[index + 1][0]
        features['next_lemma'] = tokens[index + 1][1]
        features['next_pos'] = tokens[index + 1][2]
    else:
        features['next_word'] = 'END'
    
    features['bigram'] = f"{features['prev_word']} {word}"
    if index < len(tokens) - 1:
        features['trigram'] = f"{features['prev_word']} {word} {features['next_word']}"
    
    # Add window-based features
    if index > 1:
        features['prev_word2'] = tokens[index - 2][0]
    else:
        features['prev_word2'] = 'START2'

    if index < len(tokens) - 2:
        features['next_word2'] = tokens[index + 2][0]
    else:
        features['next_word2'] = 'END2'
    
    return features

def prepare_data(annotated_data, word_frequencies, sentiment_analyzer):
    prepared_data = []
    for i in range(len(annotated_data)):
        features = extract_features(i, annotated_data, word_frequencies, sentiment_analyzer)
        label = annotated_data[i][4]  # Updated to account for the new position of the label
        prepared_data.append((features, label))
    return prepared_data

def get_word_frequencies(data):
    word_freq = Counter(word.lower() for word, _, _, _, _ in data)  # Adjusted for the extra column
    return word_freq

def compute_class_weights(data, weight_factor=0.5):
    """
    Computes class weights for handling class imbalance with an adjustable weight factor.
    :param data: List of annotated data with labels.
    :param weight_factor: Factor to scale the computed class weights (default is 1.0).
    :return: Dictionary of class weights.
    """
    # Count the occurrences of each label
    label_counts = Counter(label for _,_,_,_, label in data)
    total_count = len(data)

    # Calculate weights as the inverse of class frequency
    raw_class_weights = {label: total_count / count for label, count in label_counts.items()}

    # Apply the weight factor to adjust the impact of class weights
    class_weights = {label: weight ** weight_factor for label, weight in raw_class_weights.items()}
    
    print(f"Raw class weights (without factor): {raw_class_weights}")
    print(f"Adjusted class weights (with factor {weight_factor}): {class_weights}")
    
    return class_weights
def train_maxent_classifier(training_data, class_weights):
    if not training_data:
        raise ValueError("Training data is empty")
    
    print(f"Number of training samples: {len(training_data)}")
    labels = set(label for _, label in training_data)
    print(f"Unique labels in training data: {labels}")

    # Apply class weights by duplicating samples
    weighted_data = []
    for features, label in training_data:
        weight = int(class_weights[label])
        weighted_data.extend([(features, label)] * weight)

    print(f"Weighted training samples: {len(weighted_data)}")

    try:
        model = MaxentClassifier.train(weighted_data, algorithm='iis', max_iter=10)
        return model
    except Exception as e:
        print(f"Error during training: {e}")
        raise
def predict_and_save(model, input_file, output_file, sentiment_analyzer, word_frequencies):
    print(f"Reading from {input_file}")
    print(f"Writing to {output_file}")
    predictions_made = 0

    try:
        with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
            reader = csv.reader(infile, delimiter='\t')
            writer = csv.writer(outfile, delimiter='\t')

            for line_num, row in enumerate(reader, 1):
                if not row:
                    writer.writerow([])  # Write a blank line to keep the same line structure
                    continue

                if len(row) < 3:
                    print(f"Warning: Skipping incomplete line {line_num}")
                    writer.writerow(row)  # Write the original incomplete line as-is
                    continue

                word, lemma, pos, sentence = row[:4]  # Account for the sentence column
                token_data = [(word, lemma, pos, sentence, None)]
                features = extract_features(0, token_data, word_frequencies, sentiment_analyzer)  # index 0 since there's only one token
                prediction = model.classify(features)

                writer.writerow([word, lemma, pos, sentence, prediction])  # Include the sentence column in the output
                predictions_made += 1

                if predictions_made <= 5:
                    print(f"Sample prediction {predictions_made}: {row} -> {prediction}")

        print(f"Total predictions made: {predictions_made}")
    except Exception as e:
        print(f"Error during prediction: {e}")

def peek_file(file_path, num_lines=5):
    print(f"First {num_lines} lines of {file_path}:")
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                print(f"  {i+1}: {line.strip()}")
    except Exception as e:
        print(f"Error peeking into file {file_path}: {e}")

if __name__ == "__main__":
    train_file = 'train.pos'
    dev_file = 'dev_to_annotate.pos'
    output_file = 'dev_annotated.pos'

    print(f"Current working directory: {os.getcwd()}")
    print("Files in current directory:")
    print("\n".join(os.listdir('.')))

    if not os.path.exists(train_file):
        print(f"Error: {train_file} does not exist in the current directory.")
    else:
        print(f"\nPeeking into {train_file}:")
        peek_file(train_file)

        print("\nLoading training data...")
        train_data = load_data(train_file)
        
        if train_data:
            print(f"\nTotal samples in training data: {len(train_data)}")
            word_frequencies = get_word_frequencies(train_data)
            sentiment_analyzer = SentimentIntensityAnalyzer()

            training_data = prepare_data(train_data, word_frequencies, sentiment_analyzer)
            print(f"Prepared training samples: {len(training_data)}")
            
            # Compute class weights
            class_weights = compute_class_weights(train_data)

            print("\nTraining the model with class weights...")
            try:
                model = train_maxent_classifier(training_data, class_weights)
                print("Model training completed successfully")

                print(f"\nPredicting on {dev_file}...")
                if os.path.exists(dev_file):
                    predict_and_save(model, dev_file, output_file, sentiment_analyzer, word_frequencies)
                    # Add this after the predict_and_save function call
                    if os.path.exists(output_file):
                        file_size = os.path.getsize(output_file)
                        print(f"Output file size: {file_size} bytes")
                        if file_size > 0:
                            print("Output file is not empty")
                            with open(output_file, 'r') as f:
                                print("First few lines of output file:")
                                for i, line in enumerate(f):
                                    if i >= 5:
                                        break
                                    print(line.strip())
                        else:
                            print("Warning: Output file is empty")
                    else:
                        print("Error: Output file was not created")
                    print(f"Predictions saved to {output_file}")
                    print(f"Use the scoring script to compare {output_file} with dev_gold.pos")
                else:
                    print(f"Error: {dev_file} does not exist in the current directory.")
            except Exception as e:
                print(f"Failed to train the model or make predictions: {e}")
        else:
            print("Warning: No training data available")
