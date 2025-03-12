import argparse
import pandas as pd
from misclassification_extractor import run_misclassification_extraction
from synonym_replacer import update_dataframe_texts
from data_augmentation import balance_dataset
from classification_pipeline import run_classification

def main():
    parser = argparse.ArgumentParser(
        description="DeBERTa Text Classification with Automated Synonym Replacement and Data Augmentation for Balancing"
    )
    parser.add_argument("--csv_file", type=str, required=True,
                        help="Path to the CSV dataset file (e.g. original_dataset.csv)")
    parser.add_argument("--synonyms", type=str, default="false",
                        help="Set to 'true' to perform synonym replacement based on misclassified common words (default: false)")
    parser.add_argument("--balance", type=str, default="false",
                        help="Set to 'true' to perform data augmentation for balancing (default: false)")
    args = parser.parse_args()
    
    # Load the original dataset
    df = pd.read_csv(args.csv_file)
    
    # Run misclassification-based synonym replacement if enabled.
    if args.synonyms.lower() == "true":
        print("Synonym replacement enabled. Running two iterations of misclassification analysis and replacement...")
        common_words = run_misclassification_extraction(df, threshold=0.2, epochs=10)
        df = update_dataframe_texts(df, common_words)
        print("First iteration complete.")
        common_words = run_misclassification_extraction(df, threshold=0.2, epochs=10)
        df = update_dataframe_texts(df, common_words)
        print("Second iteration complete.")
    else:
        print("Synonym replacement disabled. Proceeding with the original dataset.")
    
    # Run data augmentation for balancing if enabled.
    if args.balance.lower() == "true":
        print("Data augmentation for balancing enabled. Balancing dataset...")
        df = balance_dataset(df)
        print("Dataset balancing complete.")
    else:
        print("Dataset balancing disabled. Proceeding with current dataset.")
    
    # Run the main classification pipeline using the (updated) DataFrame.
    run_classification(df)

if __name__ == "__main__":
    main()