from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack
import nltk

# Load NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

def get_context_features(row):
    # Define context feature extraction logic
    return " ".join(row['Sentence'].split())

def get_lemma_and_pos(word):
    # Define lemma and POS tagging logic
    return f"{word}_POS"

def get_pos_tags(row):
    # Define context POS tagging logic
    return "POS_TAGS"

def extract_features(data):
    print("Extracting context features...")
    data['Context_Features'] = data.apply(get_context_features, axis=1)

    print("Extracting target features...")
    data['Target_Features'] = data['Word'].apply(get_lemma_and_pos)

    print("Extracting context POS features...")
    data['Context_POS'] = data.apply(get_pos_tags, axis=1)

    print("Building feature matrix...")
    vectorizer_word = TfidfVectorizer(max_features=5000)
    vectorizer_context = TfidfVectorizer(max_features=5000)
    vectorizer_pos = CountVectorizer(max_features=1000)

    X_word = vectorizer_word.fit_transform(data['Target_Features'])
    X_context = vectorizer_context.fit_transform(data['Context_Features'])
    X_pos = vectorizer_pos.fit_transform(data['Context_POS'])

    X = hstack([X_word, X_context, X_pos])
    y = data['Is_Metaphor_Label']
    return X, y, data

def generate_predicted_labels(data_subset, predictions, filename='predicted_labels.pos'):
    with open(filename, 'w', encoding='utf-8') as f:
        for (idx, row), pred in zip(data_subset.iterrows(), predictions):
            word = row['Word'].strip()
            lemma_pos = get_lemma_and_pos(word)
            lemma, pos = (lemma_pos.split('_') + ['UNK', 'UNK'])[:2]
            metaphor_label = 'True' if pred == 1 else 'False'
            f.write(f"{word}\t{lemma}\t{pos}\t{metaphor_label}\n")
    print(f"Predicted labels file generated: {filename}")

def generate_annotation_gold(data_subset, filename='annotation_gold.pos'):
    with open(filename, 'w', encoding='utf-8') as f:
        for idx, row in data_subset.iterrows():
            word = row['Word'].strip()
            lemma_pos = get_lemma_and_pos(word)
            lemma, pos = (lemma_pos.split('_') + ['UNK', 'UNK'])[:2]
            metaphor_label = 'True' if row['Is_Metaphor_Label'] == 1 else 'False'
            f.write(f"{word}\t{lemma}\t{pos}\t{metaphor_label}\n")
    print(f"Gold standard labels file generated: {filename}")
