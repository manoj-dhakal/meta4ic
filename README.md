# 🌟 Meta4ic: Automated Metaphor Identification in Text

## 👥 Team Members and Roles

- 💻 **Lead Programmer / Paper Writer:** Yaghyesh Ghimire
- 🗣️ **Linguistic Expert / Paper Writer:** Manoj Dhakal
- 📊 **Evaluation Specialist / Paper Writer:** Aabaran Paudel
- ✍️ **Paper Writer:** Morin Zhou

## 🎯 Problem Statement

Metaphor identification poses a significant challenge in Natural Language Processing (NLP). The Meta4ic project aims to develop a model capable of automatically identifying metaphors within text, comparing its results with human annotations. By improving our understanding of figurative language processing, this research holds potential for enhancing NLP applications, including machine translation and sentiment analysis.

## 📋 Evaluation Plan

### 🔍 Task
The project's objective is to identify metaphorical expressions within a given text corpus.

### 📏 Metrics
We will evaluate model performance through:
- 🎯 Precision
- 🔍 Recall
- 🏆 F1-score

Focusing on the balance between accuracy and the model's capacity to detect subtle figurative language.

### 📚 Dataset
We will either create or utilize an existing metaphor-annotated dataset. This process will involve gathering a diverse range of metaphorical language examples and ensuring the data accurately represents various genres of text.

### 🧠 Human Comparison
We will conduct human annotation by enlisting linguistic experts to manually label metaphors. These results will then serve as a benchmark for comparing the model's effectiveness and for performing in-depth error analysis.

## 📖 Related Work

1. 📑 ["Metaphor Identification in Large Texts Corpora"](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0062343) (Neuman et al., 2013)
   - 💡 Relevance: Provides algorithms for automatic metaphor detection and offers a comparison to human judgment.

2. 🔗 ["Exploring Chain-of-Thought for Multi-modal Metaphor Identification"](https://openreview.net/forum?id=VonVCtRipL8) (Anonymous, 2024)
   - 💡 Relevance: Explores knowledge transfer from larger language models to smaller ones.

3. 🧠 ["Evaluating the Prediction Model of Metaphor Comprehension"](https://link.springer.com/article/10.3758/s13428-021-01558-w) (Reid & Katz, 2021)
   - 💡 Relevance: Provides insights into model performance relative to human judgments.

4. 🛠️ ["Design and Evaluation of Metaphor Processing Systems"](https://direct.mit.edu/coli/article/41/4/579/1515/Design-and-Evaluation-of-Metaphor-Processing) (Shutova, 2015)
   - 💡 Relevance: Discusses system features and evaluation techniques for metaphor processing.

5. 🔍 ["Explainable Metaphor Identification Inspired by Conceptual Metaphor Theory"](https://cdn.aaai.org/ojs/21313/21313-13-25326-1-2-20220628.pdf) (Anonymous, 2022)
   - 💡 Relevance: Presents methods for explainable metaphor identification with human evaluation.

## 🚀 Strategy for Solving the Problem

1. **🎯 Define Project Scope:** Identify metaphorical language within diverse poetic texts.

2. **📚 Dataset Preparation:**
   - 📜 Collect Poetry Samples
   - 🏷️ Annotate the Data
   - 💾 Format the Data
   - 🔄 Alternative: Use existing metaphor-annotated datasets

3. **🔧 Feature Engineering:**
   - 🧹 Text Preprocessing
   - 🔍 Feature Extraction: N-grams, POS Tags, Word Embeddings, Syntactic Features

4. **🏋️ Model Training:**
   - 🔪 Dataset Splitting: 70% training, 15% validation, 15% testing
   - 🔢 Data Vectorization
   - 🤖 Model Implementation: Train a MaxEnT model

5. **📊 Model Evaluation:**
   - ✅ Validation
   - 🧪 Testing
   - 🔍 Error Analysis

6. **🔄 Refinement:** Iteratively refine feature engineering and model parameters.

7. **📝 Documentation and Reporting:** Summarize methodology, key findings, and model effectiveness.

## 🤝 Collaboration Plan

- 🗓️ Regular weekly meetings
- 👥 Roles:
  - 📚 Data Preparation: Linguistic Expert and Evaluation Specialist
  - 💻 Model Implementation: Lead Programmer
  - 📊 Evaluation: Evaluation Specialist, assisted by Lead Programmer
  - ✍️ Reporting: Paper Writer, with contributions from each team member
- 💬 Communication via Slack and GitHub
- ⏱️ Project timeline with regular milestones

## 🔮 Further Work

Extend the approach to recognize metaphorical expressions spanning multiple words, such as idioms or phrases.

## 🔍 Identification Process

Our approach follows the Metaphor Identification Procedure Vrije Universiteit (MIPVU):

1. 📖 Determine Contextual Meaning
2. 🔎 Identify Basic Meaning
3. ⚖️ Check for Contrast
4. 🏷️ Assess Metaphoricity

### 💡 Example Analysis

Sentence: "The idea blossomed in her mind."
- 🌸 Basic Meaning: Physical flowering process
- 💭 Contextual Meaning: Development of an idea
- 🏷️ Metaphorical Labeling: "Blossomed" describes idea development, not physical flowering

---

🌟 **Join us in revolutionizing metaphor identification in NLP!** 🚀
