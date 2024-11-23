import sys

def score_predictions(annotated_file, gold_file):
    with open(annotated_file, 'r') as f:
        annotated_lines = f.readlines()
    
    with open(gold_file, 'r') as f:
        gold_lines = f.readlines()

    if len(annotated_lines) != len(gold_lines):
        raise ValueError('Annotated and gold files must have the same number of lines.')

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    total = len(gold_lines)
    
    for a_line, g_line in zip(annotated_lines, gold_lines):
        a_label = a_line.strip().split('\t')[-1].lower() == 'true'
        g_label = g_line.strip().split('\t')[-1].lower() == 'true'
        
        if a_label and g_label:
            true_positives += 1
        elif a_label and not g_label:
            false_positives += 1
        elif not a_label and g_label:
            false_negatives += 1
        else:
            true_negatives += 1

    accuracy = (true_positives + true_negatives) / total * 100

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python score.py <annotated_file> <gold_file>")
        print("Example: python score.py dev_annotated.pos dev_gold.pos")
        sys.exit(1)
    
    annotated_file = sys.argv[1]
    gold_file = sys.argv[2]
    
    score_predictions(annotated_file, gold_file)
