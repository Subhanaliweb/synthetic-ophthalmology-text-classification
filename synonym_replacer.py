from nltk.corpus import wordnet

def get_synonym(word):
    """
    Returns a synonym for the given word (if one is found); otherwise, returns the original word.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower():
                synonyms.add(lemma.name().replace('_', ' '))
    return next(iter(synonyms), word)

def replace_common_words_in_text(text, common_words):
    """
    For every word in the text that appears (case-insensitively) in the common_words set,
    replace it with a synonym.
    """
    words = text.split()
    replaced_words = []
    for word in words:
        if word.lower() in common_words:
            synonym = get_synonym(word)
            replaced_words.append(synonym)
        else:
            replaced_words.append(word)
    return ' '.join(replaced_words)

def update_dataframe_texts(df, common_words_by_pair, label_column='class_label', text_column='text'):
    """
    For each significant class pair (key formatted as "class1 - class2"), update the texts of the second class
    by replacing any word found in the common words set.
    """
    for key, common_words in common_words_by_pair.items():
        class1, class2 = [s.strip() for s in key.split('-')]
        lower_common_words = set(word.lower() for word in common_words)
        mask = df[label_column].str.strip() == class2
        df.loc[mask, text_column] = df.loc[mask, text_column].apply(
            lambda x: replace_common_words_in_text(x, lower_common_words)
        )
    return df