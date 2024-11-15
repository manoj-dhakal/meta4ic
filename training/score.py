# score.py

from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

def load_labels(filename):
    """
    Load label lists from a .pos file.
    """
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            fields = line.split('\t')
            if len(fields) < 4:
                continue  # Skip improperly formatted lines
            label = fields[3]
            labels.append(1 if label.lower() == 'true' else 0)
    return labels

def evaluate_predictions(y_true, y_pred):
    """
    Calculate evaluation scores based on true and predicted labels.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    print("Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def main():
    # Specify .pos file paths
    gold_file = 'annotation_gold.pos'
    predicted_file = 'predicted_labels.pos'

    # Load labels
    print("Loading gold standard labels...")
    gold_labels = load_labels(gold_file)
    print("Loading predicted labels...")
    predicted_labels = load_labels(predicted_file)

    # Check if lengths match
    if len(gold_labels) != len(predicted_labels):
        print(f"Error: Gold labels count ({len(gold_labels)}) and predicted labels count ({len(predicted_labels)}) mismatch!")
        return

    # Evaluate predictions
    print("Starting evaluation...")
    evaluate_predictions(gold_labels, predicted_labels)

if __name__ == "__main__":
    main()
