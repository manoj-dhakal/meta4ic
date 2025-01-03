import os
import csv
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
                    if is_metaphor.lower() == 'true':
                        is_metaphor = 1
                    else:
                        is_metaphor = 0
                    annotated_data.append((word, lemma, pos, sentence, is_metaphor))
                else:
                    print(f"Warning: Skipping malformed line {line_num}: {line.strip()}")
        
        print(f"Loaded {len(annotated_data)} tokens from {file_path}")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
    
    return annotated_data

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

    # # Previous word and POS
    # if index > 0:
    #     features['prev_word'] = tokens[index - 1][0]
    #     features['prev_pos'] = tokens[index - 1][2]
    # else:
    #     features['prev_word'] = 'START'
    #     features['prev_pos'] = 'START_POS'

    # # Next word and POS
    # if index < len(tokens) - 1:
    #     features['next_word'] = tokens[index + 1][0]
    #     features['next_pos'] = tokens[index + 1][2]
    # else:
    #     features['next_word'] = 'END'
    #     features['next_pos'] = 'END_POS'

    return features

def prepare_data(annotated_data):
    """Prepare data for training or validation."""
    prepared_data = []
    for i in range(len(annotated_data)):
        features = extract_features(i, annotated_data)
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
                # Unpack sentence tuple directly into feature_dict and label
                feature_dict, label = sentence

                word = feature_dict['word']
                lemma = feature_dict['lemma']
                pos = feature_dict['pos']
                sentence_text = feature_dict['sentence']

                # Check if the sentence has changed
                if sentence_text != current_sentence:
                    if current_sentence is not None:
                        f.write("\n")  # Insert an empty line between sentences
                    current_sentence = sentence_text

                f.write(f"{word}\t{lemma}\t{pos}\t{sentence_text}\t{label}\n")
        # Add an empty line at the end of the file
        if current_sentence is not None:
            f.write("\n")
    print(f"Output written to {output_file}")

def check_for_nans(features):
    """Check if any feature contains NaN values."""
    for feature_dict in features:
        for key, value in feature_dict.items():
            if value is None or value == 'NaN' or value != value:  # Check if value is NaN
                print(f"Warning: NaN detected in feature {key} with value {value}")
                return True
    return False

def predict_and_annotate(model, vectorizer, data, output_file):
    """Predict labels for the data and write the annotated data to a file."""
    print(f"Predicting and annotating data for {output_file}...")

    # Extract features from the data
    features = [features for features, _ in data]

    # Vectorize the features
    X = vectorizer.transform(features)

    # Predict labels
    predictions = model.predict(X)

    # Prepare the annotated data
    annotated_data = []
    for i, (feature_dict, _) in enumerate(data):
        label = predictions[i]
        annotated_data.append((feature_dict, label))

    # Write the annotated data to the output file
    write_pos_file(output_file, annotated_data)

    print(f"Annotated data written to {output_file}")

if __name__ == "__main__":
    # File paths
    input_file = "final_dataset.pos"
    dev_annotated_file = "dev_annotated_linear_boost.pos"
    dev_gold_file = "dev_gold_linear_boost.pos"

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
    train_features = [features for features, label in train_data]
    train_labels = [label for features, label in train_data]
 
    dev_data = prepare_data(dev_data)
    dev_features = [features for features, label in dev_data]
    dev_labels = [label for features, label in dev_data]
    
    print("Unique labels in training data:", set(train_labels))
    
    if check_for_nans(train_features):
        print("Warning: NaNs detected in training features.")
    if check_for_nans(dev_features):
        print("Warning: NaNs detected in development features.")
        
    # Vectorize features
    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(train_features)
    X_dev = vectorizer.transform(dev_features)

    from xgboost import XGBClassifier

    # Train the XGBoost model
    print("Training the XGBoost model...")
    model = XGBClassifier(n_estimators=1500, random_state=42, learning_rate=0.7, max_depth=8)
    model.fit(X_train, train_labels)

    # Evaluate on dev set
    print("Evaluating on dev set...")
    dev_predictions = model.predict(X_dev)
    print(classification_report(dev_labels, dev_predictions, target_names=['False', 'True']))

    # Prepare and save dev gold data
    print("Saving dev gold data...")
    write_pos_file(dev_gold_file, dev_data)

    # Predict and save dev annotated data
    print("Predicting and saving dev annotated data...")
    predict_and_annotate(model, vectorizer, dev_data, dev_annotated_file)

    print("\nProcess completed.")
    print(f"Outputs generated:\n - {dev_annotated_file}\n - {dev_gold_file}")