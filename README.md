# Evaluating Synthetic Ophthalmology Datasets for Text Classification: A Study on LLM-Generated Medical Reports

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular Python framework for evaluating synthetic ophthalmology reports through advanced NLP techniques. Developed as part of TU Dresden's research on LLM-generated medical data quality assessment.

**Research Focus**: Evaluating the clinical validity and classification utility of synthetic diabetic macular edema (DME) reports generated by large language models (LLMs). Assesses diversity, terminology accuracy, and structured data adherence through a novel text classification pipeline.

**Key Contributions**:
- Clinical relevance evaluation framework
- Synthetic data quality metrics
- Adaptive classification pipeline for medical texts

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Modules](#modules)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)

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
   git clone https://github.com/Subhanaliweb/synthetic-ophthalmology-text-classification.git
   cd synthetic-ophthalmology-text-classification
   ```
2. Install the Required Dependencies:
   ```bash
   pip install torch transformers scikit-learn nltk seaborn matplotlib datasets
   ```
2. Download NLTK Data:
   Open a Python shell or add this to your code to download necessary NLTK data:
   ```
   import nltk
   nltk.download('wordnet')
   nltk.download('stopwords')
   ```

## Usage
   The application can be run from the command line and accepts several parameters:
   
#### Command-Line Parameters

- **--csv_file** (required):  
  The path to your CSV dataset file. Your CSV must include at least the following columns:
  - ```text```: The text content.
  - ```class_label```: The class label for each sample.
 
- **--synonyms** (optional, default: false): 
  Set to ```true``` to perform two iterations of misclassification-based synonym replacement. This process analyzes misclassified class pairs, extracts common words, and replaces them with synonyms.

- **--balance** (optional, default: false):
  Set to ```true``` to enable data augmentation for balancing the dataset. This will augment minority classes using synonym replacement until each class reaches the target sample count.
  
#### Example Command
   ```bash
   python main.py --csv_file path/to/your/dataset.csv --synonyms true --balance true
   ```

When you run the above command, the application will:

1. **Perform Synonym Replacement:**  
   Run two iterations of misclassification analysis and synonym replacement on the dataset.
   
2. **Perform Data Augmentation** (Optional):  
   Balance the dataset by augmenting minority classes if the ```--balance``` flag is set to ```true```.

3. **Run the Classification Pipeline:**  
   Use the processed (and possibly balanced) dataset to train and evaluate a DeBERTa-based classifier.

For additional help, run:
```bash
python main_app.py --help
```
This will display detailed descriptions of all command-line arguments.

