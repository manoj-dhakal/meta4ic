import os
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.classify import MaxentClassifier
import nltk

# Ensure required NLTK resources are available
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

def load_pos_data(file_path):
    """Load data from a .pos file."""
    annotated_data = []
    print(f"Attempting to load data from: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Split by tab characters
                parts = line.strip().split('\t')
                
                if len(parts) == 5:  # Expecting 5 columns: Word, Lemma, POS, Sentence, Is_Metaphor
                    word, lemma, pos, sentence, is_metaphor = parts
                    annotated_data.append((word, lemma, pos, sentence, is_metaphor))
                else:
                    print(f"Warning: Skipping malformed line {line_num}: {line.strip()}")
        
        print(f"Loaded {len(annotated_data)} tokens from {file_path}")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
    
    return annotated_data
    


import nltk
from nltk.corpus import stopwords

def extract_features(index, tokens):
    """Extract features for a token at a specific index."""
    word, lemma, pos, sentence, _ = tokens[index]
   
    # Ensure no feature is empty or None
    if not word or not lemma or not pos or not sentence:
        print(f"Warning: Missing feature for index {index} - word: {word}, lemma: {lemma}, pos: {pos}, sentence: {sentence}")
        return None  # Return None or handle appropriately if a feature is missing

    features = {
        'word': word,
        'lemma': lemma,
        'pos': pos,
        'sentence': sentence,
    }

    # Previous word and POS
    if index > 0:
        features['prev_word'] = tokens[index - 1][0]
        features['prev_pos'] = tokens[index - 1][2]
    else:
        features['prev_word'] = 'START'
        features['prev_pos'] = 'START_POS'

    # Next word and POS
    if index < len(tokens) - 1:
        features['next_word'] = tokens[index + 1][0]
        features['next_pos'] = tokens[index + 1][2]
    else:
        features['next_word'] = 'END'
        features['next_pos'] = 'END_POS'

    return features

def prepare_data(annotated_data):
    """Prepare data for training or validation."""
    prepared_data = []
    for i in range(len(annotated_data)):
        features = extract_features(i, annotated_data)
       
        #checking if the length of features is not 4
        if features is None:  # Skip invalid feature data
            continue
        
        label = annotated_data[i][4]
    
        prepared_data.append((features, label))
    print(f"Prepared {len(prepared_data)} samples.")
    return prepared_data


def write_pos_file(output_file, data):
    """Write predictions or ground truth to a .pos file."""
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        current_sentence = None
        for sentence in data:
            if sentence:  # Write non-empty sentences
                feature_dict, label = sentence
                word = feature_dict['word']
                lemma = feature_dict['lemma']
                pos = feature_dict['pos']
                sentence_text = feature_dict['sentence']
                if sentence_text != current_sentence:
                    if current_sentence is not None:
                        f.write("\n")  # Insert an empty line between sentences
                    current_sentence = sentence_text

                f.write(f"{word}\t{lemma}\t{pos}\t{sentence_text}\t{label}\n")
        if current_sentence is not None:
            f.write("\n")
    print(f"Output written to {output_file}")


if __name__ == "__main__":
    # File paths
    input_file = "final_dataset.pos"
    dev_annotated_file = "dev_annotated_maxent.pos"
    dev_gold_file = "dev_gold_maxent.pos"

    # Load data
    print("Loading data...")
    data = load_pos_data(input_file)
    if not data:
        print("Error: No data loaded.")
        exit(1)
    print(f"Loaded {len(data)} sentences.")

    # Split into training and dev sets (80/20 split, no shuffle for consistency)
    train_data, dev_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=False)

    # Prepare training data
    print("Preparing training data...")
    train_data = prepare_data(train_data)

    # Extract features and labels from train_data and dev_data
    train_features_labels = [(features, label) for features, label in train_data]
    dev_data = prepare_data(dev_data)
    dev_features_labels = [(features, label) for features, label in dev_data]
    print("SAMPLE DATA:", train_features_labels[50][1])
    train_labels = [label for _, label in train_data]
    print("Unique labels in training data:", set(train_labels))
   

    # Train the MaxEnt model using NLTK's MaxentClassifier
    print("Training the MaxEnt model...")
    maxent_model = MaxentClassifier.train(
        train_features_labels,
        algorithm='iis',  # Improved Iterative Scaling
        max_iter=10
    )

    # Evaluate on dev set
    print("Evaluating on dev set...")
    dev_features = [features for features, _ in dev_features_labels]
    dev_labels = [label for _, label in dev_features_labels]
    dev_predictions = [maxent_model.classify(features) for features in dev_features]
    print(classification_report(dev_labels, dev_predictions, target_names=['False', 'True']))

    # Save dev gold data
    print("Saving dev gold data...")
    write_pos_file(dev_gold_file, dev_features_labels)

    # Annotate and save dev annotated data
    print("Predicting and saving dev annotated data...")
    annotated_data = [(features, maxent_model.classify(features)) for features, _ in dev_features_labels]
    write_pos_file(dev_annotated_file, annotated_data)

    print("\nProcess completed.")
    print(f"Outputs generated:\n - {dev_annotated_file}\n - {dev_gold_file}")