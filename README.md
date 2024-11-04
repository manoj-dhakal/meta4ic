# Meta4ic: Automated Metaphor Identification in Text

## Team Members and Roles

- **Lead Programmer / Paper Writer:** Yaghyesh Ghimire
- **Linguistic Expert / Paper Writer:** Manoj Dhakal
- **Evaluation Specialist / Paper Writer:** Aabaran Paudel
- **Paper Writer:** Morin Zhou

## Problem Statement

Metaphor identification poses a significant challenge in Natural Language Processing (NLP). The Meta4ic project aims to develop a model capable of automatically identifying metaphors within text, comparing its results with human annotations. By improving our understanding of figurative language processing, this research holds potential for enhancing NLP applications, including machine translation and sentiment analysis.

## Evaluation Plan

### Task
The project's objective is to identify metaphorical expressions within a given text corpus.

### Metrics
We will evaluate model performance through Precision, Recall, and F1-score, focusing on the balance between accuracy and the model's capacity to detect subtle figurative language.

### Dataset
We will either create or utilize an existing metaphor-annotated dataset. This process will involve gathering a diverse range of metaphorical language examples and ensuring the data accurately represents various genres of text.

### Human Comparison
We will conduct human annotation by enlisting linguistic experts to manually label metaphors. These results will then serve as a benchmark for comparing the model's effectiveness and for performing in-depth error analysis.

## Related Work

1. ["Metaphor Identification in Large Texts Corpora"](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0062343) (Neuman et al., 2013)
   - Relevance: Provides algorithms for automatic metaphor detection and offers a comparison to human judgment, serving as a foundation for our model's evaluation methods.

2. ["Exploring Chain-of-Thought for Multi-modal Metaphor Identification"](https://openreview.net/forum?id=VonVCtRipL8) (Anonymous, 2024)
   - Relevance: Explores a framework that enriches smaller models through knowledge transfer from larger language models, which may inform our feature engineering and model refinement steps.

3. ["Evaluating the Prediction Model of Metaphor Comprehension"](https://link.springer.com/article/10.3758/s13428-021-01558-w) (Reid & Katz, 2021)
   - Relevance: Provides insights into model performance relative to human judgments, particularly valuable for our comparative evaluation approach.

4. ["Design and Evaluation of Metaphor Processing Systems"](https://direct.mit.edu/coli/article/41/4/579/1515/Design-and-Evaluation-of-Metaphor-Processing) (Shutova, 2015)
   - Relevance: Discusses system features and evaluation techniques essential for metaphor processing, aiding in the design of our evaluation metrics.

5. ["Explainable Metaphor Identification Inspired by Conceptual Metaphor Theory"](https://cdn.aaai.org/ojs/21313/21313-13-25326-1-2-20220628.pdf) (Anonymous, 2022)
   - Relevance: Presents methods for explainable metaphor identification, with an emphasis on human evaluation, guiding our approach to interpretable model outputs.

## Strategy for Solving the Problem

1. **Define Project Scope:** Identify metaphorical language within diverse poetic texts, focusing on both individual and multi-word metaphors.

2. **Dataset Preparation:**
   - Collect Poetry Samples: Compile a broad corpus of poetry, ensuring diversity in style and metaphor use.
   - Annotate the Data: Use an annotation scheme to label metaphors within the corpus, utilizing tools such as BRAT or Prodigy.
   - Format the Data: Organize the dataset into structured formats (e.g., CSV or JSON) with metadata fields.
   - Alternative: Use existing metaphor-annotated datasets ([VU Amsterdam Metaphor Corpus](https://aclanthology.org/L16-1668/), [Metaphorical Connections](https://paperswithcode.com/dataset/metaphorical-connections))

3. **Feature Engineering:**
   - Text Preprocessing: Lowercasing, punctuation removal, and tokenization.
   - Feature Extraction: N-grams, POS Tags, Word Embeddings, Syntactic Features.

4. **Model Training:**
   - Dataset Splitting: 70% training, 15% validation, 15% testing.
   - Data Vectorization: CountVectorizer for n-grams, TfidfVectorizer for term frequency-inverse document frequency.
   - Model Implementation: Train a MaxEnT model.

5. **Model Evaluation:**
   - Validation: Optimize hyperparameters on the validation set.
   - Testing: Compute accuracy, precision, recall, and F1-score on the test set.
   - Error Analysis: Analyze misclassified examples.

6. **Refinement:** Iteratively refine feature engineering and model parameters based on evaluation results.

7. **Documentation and Reporting:** Document all steps and prepare a final report summarizing methodology, key findings, and the model's effectiveness.

## Collaboration Plan

- Regular weekly meetings
- Roles:
  - Data Preparation: Linguistic Expert and Evaluation Specialist
  - Model Implementation: Lead Programmer
  - Evaluation: Evaluation Specialist, assisted by Lead Programmer
  - Reporting: Paper Writer, with contributions from each team member
- Communication via Slack and GitHub
- Project timeline with regular milestones for progress checks

## Further Work

Extend the approach to recognize metaphorical expressions spanning multiple words, such as idioms or phrases, avoiding isolated metaphor identifications for expressions like "apple of my eye."

## Identification Process

Our approach follows the Metaphor Identification Procedure Vrije Universiteit (MIPVU):

1. Determine Contextual Meaning: Analyze each word based on the sentence's context.
2. Identify Basic Meaning: Refer to dictionary meanings, focusing on literal or concrete senses.
3. Check for Contrast: Determine if there's a distinct contrast between basic and contextual meanings.
4. Metaphoricity: If the basic meaning differs yet makes sense in an alternate context, it is likely metaphorical.

### Example Analysis

Sentence: "The idea blossomed in her mind."
- Basic Meaning: The literal sense of "blossomed" refers to the physical flowering process.
- Contextual Meaning: Here, "blossomed" signifies the development of an idea.
- Metaphorical Labeling: The use of "blossomed" to describe an idea developing rather than a plant flowering demonstrates metaphorical language.
