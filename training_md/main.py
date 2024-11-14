import nltk
from nltk.classify import MaxentClassifier
import csv
import os

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
                if len(parts) == 4:
                    word, lemma, pos, label = parts
                    annotated_data.append((word, lemma, pos, label))
                else:
                    print(f"Warning: Line {line_num} does not have 4 parts: {line}")
        
        print(f"Loaded {len(annotated_data)} samples from {file_path}")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
    
    return annotated_data

def extract_features(index, tokens):
    word, lemma, pos, _ = tokens[index]
    features = {
        'word': word,
        'lemma': lemma,
        'pos': pos,
        'is_capitalized': word[0].isupper(),
        'word_length': len(word),
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
    return features

def prepare_data(annotated_data):
    prepared_data = []
    for i in range(len(annotated_data)):
        features = extract_features(i, annotated_data)
        label = annotated_data[i][3]  # Is_Metaphor label
        prepared_data.append((features, label))
    return prepared_data

def train_maxent_classifier(training_data):
    if not training_data:
        raise ValueError("Training data is empty")
    
    print(f"Number of training samples: {len(training_data)}")
    
    labels = set(label for _, label in training_data)
    print(f"Unique labels in training data: {labels}")
    
    if not labels:
        raise ValueError("No valid labels found in training data")
    
    try:
        model = MaxentClassifier.train(training_data, algorithm='iis', max_iter=1)
        return model
    except Exception as e:
        print(f"Error during training: {e}")
        raise
import csv

def predict_and_save(model, input_file, output_file):
    print(f"Reading from {input_file}")
    print(f"Writing to {output_file}")
    predictions_made = 0

    try:
        with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
            reader = csv.reader(infile, delimiter='\t')
            writer = csv.writer(outfile, delimiter='\t')

            for line_num, row in enumerate(reader, 1):
                # Check for blank line
                if not row:
                    writer.writerow([])  # Write a blank line to keep the same line structure
                    continue

                # Ensure the row has at least word, lemma, and POS columns
                if len(row) < 3:
                    print(f"Warning: Skipping incomplete line {line_num}")
                    writer.writerow(row)  # Write the original incomplete line as-is
                    continue

                # Unpack the fields for the token
                word, lemma, pos = row[:3]
                
                # Create a temporary tokens list for context-aware feature extraction
                token_data = [(word, lemma, pos, None)]
                features = extract_features(0, token_data)  # index 0 since there's only one token

                # Make prediction using the model
                prediction = model.classify(features)

                # Append the prediction as the new label column in the row
                writer.writerow([word, lemma, pos, prediction])
                predictions_made += 1

                # Print first few predictions for debugging
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
            training_data = prepare_data(train_data)
            print(f"Prepared training samples: {len(training_data)}")
            
            print("\nSample training data:")
            for i, (features, label) in enumerate(training_data[:5]):
                print(f"Sample {i+1}:")
                print(f"  Features: {features}")
                print(f"  Label: {label}")

            print("\nTraining the model...")
            try:
                model = train_maxent_classifier(training_data)
                print("Model training completed successfully")

                print(f"\nPredicting on {dev_file}...")
                if os.path.exists(dev_file):
                    predict_and_save(model, dev_file, output_file)
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