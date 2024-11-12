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
    # Get the full sentence text
    sentence_text = ' '.join(sentence.xpath('.//tei:w//text()', namespaces=namespaces))
    
    # Process each word
    for word in sentence.xpath('.//tei:w', namespaces=namespaces):
        word_text = word.xpath('string()').strip()
        lemma = word.get('lemma')
        pos = word.get('type')
        is_metaphor = bool(word.xpath('.//tei:seg[@function="mrw"]', namespaces=namespaces))
        metaphor_type = word.xpath('.//tei:seg[@function="mrw"]/@type', namespaces=namespaces)
        metaphor_type = metaphor_type[0] if metaphor_type else None
        
        data.append({
            'Word': word_text,
            'Lemma': lemma,
            'POS': pos,
            'Is_Metaphor': is_metaphor,
            'Metaphor_Type': metaphor_type,
            'Sentence': sentence_text
        })

# Create DataFrame
df = pd.DataFrame(data)

# Display or save the DataFrame
df.to_csv("extracted_metaphors.csv", index=False)  # Save to CSV if needed
print(df.head())  # Display first few rows for verification
