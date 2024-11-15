import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet

# Load necessary resources
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

# Convert Penn Treebank POS tag to WordNet POS tag
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None  # Return None if no match

# Define a function to extract context features
def get_context_features(row):
    sentence = row['Sentence']
    word = row['Word']
    tokens = word_tokenize(sentence)
    tokens_lower = [w.lower() for w in tokens]
    word_lower = word.lower()
    if word_lower in tokens_lower:
        word_indices = [i for i, w in enumerate(tokens_lower) if w == word_lower]
        context_window = 3  # Context window size
        context_words = []
        for idx in word_indices:
            start = max(0, idx - context_window)
            end = min(len(tokens), idx + context_window + 1)
            context = tokens[start:idx] + tokens[idx+1:end]
            context_words.extend(context)
    else:
        context_words = tokens
    # Remove stop words and the target word
    context_words = [w for w in context_words if w.lower() not in stop_words and w.lower() != word_lower]
    return ' '.join(context_words)

# Extract the lemma and POS tag of the target word
def get_lemma_and_pos(word):
    try:
        tokens = word_tokenize(word)
        pos_tags = pos_tag(tokens)
        if pos_tags:
            token, tag = pos_tags[0]
            pos = get_wordnet_pos(tag)
            if pos is None:
                lemma = lemmatizer.lemmatize(token.lower())
            else:
                lemma = lemmatizer.lemmatize(token.lower(), pos=pos)
            return f"{lemma}_{tag}"
        else:
            return f"{word}_UNK"
    except Exception as e:
        # Print error message and return default value
        print(f"Error processing word '{word}': {e}")
        return f"{word}_UNK"

# Extract POS tags of the context
def get_pos_tags(row):
    sentence = row['Sentence']
    word = row['Word']
    tokens = word_tokenize(sentence)
    tokens_lower = [w.lower() for w in tokens]
    word_lower = word.lower()
    if word_lower in tokens_lower:
        word_indices = [i for i, w in enumerate(tokens_lower) if w == word_lower]
        context_window = 3
        pos_tags = pos_tag(tokens)
        context_pos = []
        for idx in word_indices:
            start = max(0, idx - context_window)
            end = min(len(tokens), idx + context_window + 1)
            context = pos_tags[start:idx] + pos_tags[idx+1:end]
            context_pos.extend([tag for word, tag in context])
    else:
        pos_tags = pos_tag(tokens)
        context_pos = [tag for word, tag in pos_tags]
    return ' '.join(context_pos)
