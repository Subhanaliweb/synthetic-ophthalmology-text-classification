# DeBERTa Text Classification with Automated Synonym Replacement and Data Augmentation

A modular Python application that leverages a DeBERTa-based classifier along with an automated preprocessing pipeline to improve text classification performance. The pipeline includes:

- **Misclassification-Based Synonym Replacement:**  
  Two consecutive iterations of misclassification analysis are performed to extract common words from misclassified class pairs, which are then replaced with synonyms.

- **Data Augmentation for Balancing:**  
  Augments minority classes using synonym replacement to balance the dataset.

- **DeBERTa-Based Classification:**  
  The final classification is performed using a DeBERTa model with weighted loss.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Modules](#modules)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project is designed to enhance text classification through two main strategies:

1. **Synonym Replacement:**  
   Based on misclassification analysis using a DeBERTa model, the application identifies and replaces common words between misclassified class pairs. This process is run twice consecutively to further refine the dataset.

2. **Data Augmentation for Balancing:**  
   The dataset is balanced by augmenting minority classes using synonym replacement so that every class reaches a similar sample count.

The processed dataset is then used to train and evaluate a DeBERTa-based classifier.

## Features

- **Automated Synonym Replacement:**  
  Two iterations of misclassification analysis extract common words and replace them with synonyms.

- **Data Augmentation for Dataset Balancing:**  
  Minority classes are augmented until they match the sample count of the majority class.

- **DeBERTa-Based Classification Pipeline:**  
  Uses a DeBERTa model with weighted loss for final classification.

- **Modular Design:**  
  Each component is separated into its own module for clarity and ease of maintenance.

## Modules

- **misclassification_extractor.py**  
  Trains a DeBERTa classifier on your dataset, computes misclassification statistics (via a confusion matrix), and extracts common words from significant misclassified class pairs.

- **synonym_replacer.py**  
  Contains functions to replace common words with synonyms based on NLTK’s WordNet.

- **data_augmentation.py**  
  Provides functions for data augmentation by balancing minority classes through synonym replacement.

- **classification_pipeline.py**  
  Implements the main text classification pipeline using a DeBERTa model.

- **main_app.py**  
  The main driver that integrates all modules. It supports command-line flags to enable synonym replacement and/or data augmentation before running the final classification.

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [Transformers](https://github.com/huggingface/transformers)
- [scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [datasets](https://github.com/huggingface/datasets)

## Installation

1. **Clone the Repository** (or copy the files into a folder):

   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository