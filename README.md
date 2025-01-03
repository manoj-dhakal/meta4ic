# Meta4ic: A Metaphor Annotation System

Meta4ic is a machine learning-based system developed to annotate metaphors in textual data. This project offers an innovative approach to identifying and analyzing metaphorical expressions, assisting researchers and linguists in better understanding the subtleties of language.

This repository accompanies the research paper **["Meta4ic: A System for Metaphor Annotation in Text"](#)** (available under request), where you can find comprehensive details about the system's design, methodology, and evaluation.

---

## Highlights
- Focused on metaphor detection and annotation.
- Built for researchers, linguists, and NLP enthusiasts.
- Provides insights into metaphorical language.

## Reproduction
- To reproduce the results run the files inside the training folder in this order:
  1) parser.py [parses the training data from VUMAC.xml]
  2) balancer.py [balances the training data]
  3) one of the main files for training
  4) score.py for the trained files

## Abstract

_This study explores an alternative approach to metaphor detection using interpretable machine learning models, challenging the prevailing reliance on complex neural network architectures. By leveraging the VU Amsterdam Metaphor Corpus (VUMAC) and implementing the MIPVU methodology, we demonstrate that simpler, transparent models can achieve competitive performance when coupled with effective data balancing techniques and careful feature engineering. Our experiments with logistic regression, maximum entropy (MaxEnt), and XGBoost models on a highly imbalanced dataset reveal significant improvements through targeted balancing strategies. Our findings suggest that interpretable models, when properly optimized, can effectively identify metaphorical language while maintaining transparency in their decision-making process - a crucial advantage for cognitive and linguistic research applications. This work contributes to the ongoing discussion about balancing model complexity with interpretability in computational linguistics, offering insights into feature engineering and data balancing strategies for metaphor detection.
For more details, please refer to the linked research paper._

---

© 2025 Manoj Dhakal. All rights reserved.
