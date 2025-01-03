import sys
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

def load_labels(filename):
    """
    Load label lists from a .pos (tab-separated) file.
    """
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            fields = line.split('\t')  # Split by tab instead of comma
            if len(fields) < 5:  # Assuming label is the 5th column
                continue  # Skip improperly formatted lines
            label = fields[4]  # The label should still be the 5th column
            if label.lower() == 'true' or label == '1':
                labels.append(1)
            else:
                labels.append(0)
        
    return labels

def evaluate_predictions(y_true, y_pred):
    """
    Calculate evaluation scores based on true and predicted labels.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    try:
        auroc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auroc = None  # Handle cases where AUROC cannot be calculated

    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if auroc is not None:
        print(f"AUROC: {auroc:.4f}")
    else:
        print("AUROC: Not applicable (possibly due to lack of positive labels in the data)")
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

def evaluate_predictions(y_true, y_pred):
    """
    Calculate evaluation scores based on true and predicted labels, including AUPRC.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    try:
        auroc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auroc = None  # Handle cases where AUROC cannot be calculated

    # Calculate AUPRC (Area Under Precision-Recall Curve)
    try:
        auprc = average_precision_score(y_true, y_pred)
    except ValueError:
        auprc = None  # Handle cases where AUPRC cannot be calculated

    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if auroc is not None:
        print(f"AUROC: {auroc:.4f}")
    else:
        print("AUROC: Not applicable (possibly due to lack of positive labels in the data)")
    
    if auprc is not None:
        print(f"AUPRC: {auprc:.4f}")
    else:
        print("AUPRC: Not applicable (possibly due to lack of positive labels in the data)")

def main():
    if len(sys.argv) != 2:
        print("Usage: python score.py <method>")
        print("Example: python score.py {maxent, regression, boosted, bert}")
        return

    method = sys.argv[1].lower()
    if method == "maxent":
        gold_file = 'dev_gold_maxent.pos'
        predicted_file = 'dev_annotated_maxent.pos'
    elif method == "regression":
        gold_file = 'dev_gold_linear.pos'
        predicted_file = 'dev_annotated_linear.pos'
    elif method == "boosted":
        gold_file = 'dev_gold_linear_boost.pos'
        predicted_file = 'dev_annotated_linear_boost.pos'
    elif method == "bert":
        gold_file = 'dev_gold_bert.pos'
        predicted_file = 'dev_annotated_bert.pos'
    else:
        print(f"Error: Unknown method '{method}'. Use 'maxent', 'regression', 'boosted', or 'bert'.")
        return

    print(f"Evaluating method: {method}")
    print(f"Gold file: {gold_file}")
    print(f"Predicted file: {predicted_file}")

    try:
        # Load labels
        print("\nLoading gold standard labels...")
        gold_labels = load_labels(gold_file)
        print(f"Loaded {len(gold_labels)} labels from {gold_file}")

        print("\nLoading predicted labels...")
        predicted_labels = load_labels(predicted_file)
        print(f"Loaded {len(predicted_labels)} labels from {predicted_file}")

        # Check if lengths match
        if len(gold_labels) != len(predicted_labels):
            print(f"Error: Gold labels count ({len(gold_labels)}) and predicted labels count ({len(predicted_labels)}) mismatch!")
            return

        # Evaluate predictions
        print("\nStarting evaluation...")
        evaluate_predictions(gold_labels, predicted_labels)
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
