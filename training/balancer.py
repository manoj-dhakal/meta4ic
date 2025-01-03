import numpy as np

def count_metaphors_per_sentence(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    sentence_dict = {}
    current_sentence = ""
    metaphor_count = 0

    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            parts = stripped_line.split('\t')
            if len(parts) == 5:
                word, ground_word, pos, sentence, label = parts
                if sentence != current_sentence:
                    if current_sentence:
                        sentence_dict[current_sentence] = metaphor_count
                    current_sentence = sentence
                    metaphor_count = 0
                if label.lower() == 'true':
                    metaphor_count += 1
        elif current_sentence:
            sentence_dict[current_sentence] = metaphor_count
            current_sentence = ""
            metaphor_count = 0

    # Check if the last sentence was not followed by an empty line
    if current_sentence:
        sentence_dict[current_sentence] = metaphor_count

    return sentence_dict

# File path
file_path = 'extracted_metaphors.pos'

# Count metaphors per sentence
metaphor_counts = count_metaphors_per_sentence(file_path)

# Count total number of sentences
total_sentences = len(metaphor_counts)

# Count number of sentences with 0 metaphors
zero_metaphor_sentences = sum(1 for count in metaphor_counts.values() if count == 0)

# Print the counts
print(f'Total number of sentences: {total_sentences}')
print(f'Number of sentences with 0 metaphors: {zero_metaphor_sentences}')

# Initialize counters for True and False labels
true_count = 0
false_count = 0

# Count True and False labels
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            parts = stripped_line.split('\t')
            if len(parts) == 5:
                label = parts[4].lower()
                if label == 'true':
                    true_count += 1
                elif label == 'false':
                    false_count += 1

# Calculate total labels
total_labels = true_count + false_count

# Calculate percentages
true_percentage = (true_count / total_labels) * 100 if total_labels > 0 else 0
false_percentage = (false_count / total_labels) * 100 if total_labels > 0 else 0

# Print the percentages
print(f'Percentage of True labels: {true_percentage:.2f}%')
print(f'Percentage of False labels: {false_percentage:.2f}%')

# Remove sentences with 0 metaphors from the dictionary
filtered_sentences = {sentence: count for sentence, count in metaphor_counts.items() if count > 0}

# Find the fourth quartile
counts = list(filtered_sentences.values())
counts.sort()
#median_last_half = np.median(counts[len(counts)//2:])
fourth_quartile = np.percentile(counts, 75)

# Filter sentences in the fourth quartile
fourth_quartile_sentences = {sentence: count for sentence, count in filtered_sentences.items() if count >= fourth_quartile}

# Write the filtered and oversampled sentences to a new file
with open('final_dataset.pos', 'w') as output_file:
    with open(file_path, 'r') as input_file:
        lines = input_file.readlines()
        current_sentence = ""
        sentence_lines = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:
                parts = stripped_line.split('\t')
                if len(parts) == 5:
                    word, ground_word, pos, sentence, label = parts
                    if sentence != current_sentence:
                        if current_sentence and current_sentence in filtered_sentences:
                            output_file.write(''.join(sentence_lines))
                            output_file.write('\n')  # Add an empty line between different sentences
                            if current_sentence in fourth_quartile_sentences:
                                output_file.write(''.join(sentence_lines))
                                output_file.write('\n')  # Add an empty line between different sentences
                        current_sentence = sentence
                        sentence_lines = [line]
                    else:
                        sentence_lines.append(line)
            elif current_sentence:
                if current_sentence in filtered_sentences:
                    output_file.write(''.join(sentence_lines))
                    output_file.write('\n')  # Ensure an empty line after the last sentence
                    if current_sentence in fourth_quartile_sentences:
                        output_file.write(''.join(sentence_lines))
                        output_file.write('\n')  # Ensure an empty line after the last sentence
                current_sentence = ""
                sentence_lines = []

# Print a message indicating the process is complete
print("Filtered and oversampled sentences have been written to 'final_dataset.pos'.")

# Initialize counters for True and False labels
true_count = 0
false_count = 0

# Count True and False labels in the new file
with open('final_dataset.pos', 'r') as file:
    lines = file.readlines()
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            parts = stripped_line.split('\t')
            if len(parts) == 5:
                label = parts[4].lower()
                if label == 'true':
                    true_count += 1
                elif label == 'false':
                    false_count += 1

# Calculate total labels
total_labels = true_count + false_count

# Calculate percentages
true_percentage = (true_count / total_labels) * 100 if total_labels > 0 else 0
false_percentage = (false_count / total_labels) * 100 if total_labels > 0 else 0

# Print the percentages
print(f'Percentage of True labels in the final dataset: {true_percentage:.2f}%')
print(f'Percentage of False labels in the final dataset: {false_percentage:.2f}%')
