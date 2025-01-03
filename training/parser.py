import pandas as pd
from lxml import etree

# Load XML data from local file
file_path = "VUAMC.xml"
tree = etree.parse(file_path)

# Define namespaces
namespaces = {'tei': 'http://www.tei-c.org/ns/1.0', 'vici': 'http://www.tei-c.org/ns/VICI'}

# Extract relevant data
data = []
for sentence in tree.xpath('//tei:s', namespaces=namespaces):
    
    # Combine all words in a sentence into a single string with space separation
    sentence_text = ' '.join(sentence.xpath('.//tei:w//text()', namespaces=namespaces)).strip()
    
    # Ensure no newline or extra space in the sentence text
    sentence_text = ' '.join(sentence_text.split())
    
    # Process each word in the sentence and add the full sentence to each word
    for word in sentence.xpath('.//tei:w', namespaces=namespaces):
        word_text = word.xpath('string()').strip()
        lemma = word.get('lemma')
        pos = word.get('type')
        is_metaphor = bool(word.xpath('.//tei:seg[@function="mrw"]', namespaces=namespaces))
        
        # Append word details along with the complete sentence
        data.append({
            'Word': word_text,
            'Lemma': lemma,
            'POS': pos,
            'Is_Metaphor': is_metaphor,
            'Sentence': sentence_text  # Attach the complete sentence to each word
        })
    
    # Add a blank line entry after each sentence
    data.append({'Word': None, 'Lemma': None, 'POS': None, 'Is_Metaphor': None, 'Sentence': None})

# Create DataFrame
df = pd.DataFrame(data)

# Write to file with tab-separated values
with open("extracted_metaphors.pos", "w") as f:

    
    for _, row in df.iterrows():
        # If the row is blank (after each sentence), write a blank line
        if pd.isna(row['Word']):
            f.write("\n")
        else:
            # Write the row data with tab separation, ensuring the entire sentence is on the same line
            f.write(f"{row['Word']}\t{row['Lemma']}\t{row['POS']}\t{row['Sentence']}\t{row['Is_Metaphor']}\n")

print("File 'extracted_metaphors.pos' generated successfully.")