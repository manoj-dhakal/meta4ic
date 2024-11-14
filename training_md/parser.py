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
    # Process each word
    for word in sentence.xpath('.//tei:w', namespaces=namespaces):
        word_text = word.xpath('string()').strip()
        lemma = word.get('lemma')
        pos = word.get('type')
        is_metaphor = bool(word.xpath('.//tei:seg[@function="mrw"]', namespaces=namespaces))
        
        data.append({
            'Word': word_text,
            'Lemma': lemma,
            'POS': pos,
            'Is_Metaphor': is_metaphor
        })

    # Add a blank line entry after each sentence
    data.append({'Word': None, 'Lemma': None, 'POS': None, 'Is_Metaphor': None})

# Create DataFrame without 'Metaphor_Type' and 'Sentence' columns
df = pd.DataFrame(data)

# Write to file with tab-separated values
with open("master_data.pos", "w") as f:
    # Write header
    f.write("\n")
    
    for _, row in df.iterrows():
        # If the row is blank (after each sentence), write a blank line
        if pd.isna(row['Word']):
            f.write("\n")
        else:
            # Write the row data with tab separation
            f.write(f"{row['Word']}\t{row['Lemma']}\t{row['POS']}\t{row['Is_Metaphor']}\n")