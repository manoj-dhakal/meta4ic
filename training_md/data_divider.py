import os
def split_file(input_file, train_file, dev_gold_file, test_gold_file, dev_annotate_file, test_annotate_file):
    print(f"Reading input file: {input_file}")
    try:
        with open(input_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    print(f"Total characters read: {len(content)}")

    # Split the content into sentences (paragraphs)
    sentences = content.split('\n\n')
    print(f"Total sentences: {len(sentences)}")

    # Calculate split indices
    total_sentences = len(sentences)
    train_split = int(0.8 * total_sentences)
    dev_split = int(0.9 * total_sentences)

    # Split the sentences
    train_sentences = sentences[:train_split]
    dev_sentences = sentences[train_split:dev_split]
    test_sentences = sentences[dev_split:]

    print(f"Train sentences: {len(train_sentences)}")
    print(f"Dev sentences: {len(dev_sentences)}")
    print(f"Test sentences: {len(test_sentences)}")

    # Function to remove the last column (annotation) from each line
    def remove_last_column(sentence):
        lines = sentence.split('\n')
        return '\n'.join(['\t'.join(line.split('\t')[:-1]) for line in lines])

    # Write files with error checking
    def write_file(filename, data):
        try:
            with open(filename, 'w') as f:
                f.write(data)
            print(f"Successfully wrote {filename}")
            print(f"File size: {os.path.getsize(filename)} bytes")
        except Exception as e:
            print(f"Error writing {filename}: {e}")

    # Write all files
    write_file(train_file, '\n\n'.join(train_sentences))
    write_file(dev_gold_file, '\n\n'.join(dev_sentences))
    write_file(test_gold_file, '\n\n'.join(test_sentences))
    write_file(dev_annotate_file, '\n\n'.join(remove_last_column(sentence) for sentence in dev_sentences))
    write_file(test_annotate_file, '\n\n'.join(remove_last_column(sentence) for sentence in test_sentences))

    # Check content of annotate files
    for file in [dev_annotate_file, test_annotate_file]:
        try:
            with open(file, 'r') as f:
                content = f.read()
            print(f"\nFirst 200 characters of {file}:")
            print(content[:200])
        except Exception as e:
            print(f"Error reading {file}: {e}")

# Call the function with your file names
split_file('master_data.pos', 'train.pos', 'dev_gold.pos', 'test_gold.pos', 'dev_to_annotate.pos', 'test_to_annotate.pos')