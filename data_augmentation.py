import pandas as pd
import random
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is downloaded (quietly)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

def synonym_replacement(text, n):
    """
    Replaces `n` random words in the text with synonyms using WordNet.
    """
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    augmented_text = words.copy()
    
    for _ in range(n):
        word_idx = random.choice(range(len(words)))
        word = words[word_idx]
        if len(word) <= 3:
            continue
        synonyms = wordnet.synsets(word)
        if not synonyms:
            continue
        synonym_word = random.choice(synonyms).lemmas()[0].name()
        augmented_text[word_idx] = lemmatizer.lemmatize(synonym_word)
    return ' '.join(augmented_text)

def balance_dataset(df, target_count=None):
    """
    Augments minority classes using synonym replacement so that every class has
    a total number of samples equal to `target_count` (if not provided, the max class count is used).
    """
    class_counts = df['class_label'].value_counts()
    
    if target_count is None:
        target_count = class_counts.max()
        
    balanced_data = []
    
    for class_label in class_counts.index:
        class_samples = df[df['class_label'] == class_label].copy()
        current_count = len(class_samples)
        
        if current_count >= target_count:
            balanced_data.append(class_samples.head(target_count))
            continue
        
        balanced_data.append(class_samples)
        samples_needed = target_count - current_count
        
        augmented_samples = []
        while len(augmented_samples) < samples_needed:
            sample = class_samples.sample(n=1).iloc[0]
            augmented_text = synonym_replacement(sample['text'], n=5)
            augmented_samples.append({
                'text': augmented_text,
                'class_label': class_label
            })
        
        balanced_data.append(pd.DataFrame(augmented_samples))
    
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    
    print("\nClass distribution after balancing:")
    print(balanced_df['class_label'].value_counts())
    
    return balanced_df