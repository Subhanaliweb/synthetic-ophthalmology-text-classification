import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from transformers import DebertaTokenizer, DebertaForSequenceClassification, AdamW
import torch.nn as nn
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix
import numpy as np

# --- Preprocessing ---
def preprocess_text(text):
    """Remove special characters, lowercase, remove stopwords, and stem."""
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    processed_words = [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    return ' '.join(processed_words)

# --- Dataset Class ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# --- Custom DeBERTa Classifier ---
class CustomDebertaClassifier(nn.Module):
    def __init__(self, base_model_name='microsoft/deberta-base', num_labels=2, dropout_rate=0.3):
        super(CustomDebertaClassifier, self).__init__()
        self.model = DebertaForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

# --- Training and Analysis Functions ---
def train_model(df, num_labels, epochs=10, batch_size=32, seed=42):
    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['class_label'])
    # Preprocess text and store in a new column
    df['processed_text'] = df['text'].apply(preprocess_text)
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    texts = df['processed_text'].tolist()
    labels = df['label_encoded'].tolist()
    dataset = TextDataset(texts, labels, tokenizer)
    # Split dataset into train/val/test (80/10/10)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomDebertaClassifier(num_labels=num_labels)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            loss, _ = model(input_ids, attention_mask, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} training complete.")
    # Evaluate on test set
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            _, logits = model(input_ids, attention_mask, labels_batch)
            preds = torch.argmax(logits, dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
    cm = confusion_matrix(all_labels, all_predictions)
    return cm, label_encoder, df

def compute_misclassification_rates(cm, label_encoder):
    num_classes = len(label_encoder.classes_)
    misclassification_rates = {}
    self_classification_rates = {}
    for i in range(num_classes):
        total = cm[i, :].sum()
        self_rate = cm[i, i] / total if total > 0 else 0
        self_classification_rates[label_encoder.inverse_transform([i])[0]] = self_rate
        for j in range(num_classes):
            if i != j:
                key = f"{label_encoder.inverse_transform([i])[0]} - {label_encoder.inverse_transform([j])[0]}"
                rate = cm[i, j] / total if total > 0 else 0
                misclassification_rates[key] = rate
    return misclassification_rates, self_classification_rates

def extract_significant_pairs(misclassification_rates, threshold=0.2):
    significant_pairs = {}
    for key, rate in misclassification_rates.items():
        if rate >= threshold:
            class1, class2 = key.split(' - ')
            if class1 not in significant_pairs:
                significant_pairs[class1] = []
            significant_pairs[class1].append(class2)
    return significant_pairs

def extract_common_words(df, significant_pairs):
    # Tokenize text (using similar preprocessing as above)
    def tokenize(text):
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        return [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    from collections import defaultdict
    texts_by_class = defaultdict(list)
    for _, row in df.iterrows():
        tokens = tokenize(row['text'])
        texts_by_class[row['class_label']].extend(tokens)
    common_words_by_pair = {}
    for main_class, misclasses in significant_pairs.items():
        main_words = set(texts_by_class[main_class])
        for misclass in misclasses:
            misclass_words = set(texts_by_class[misclass])
            common_words = main_words.intersection(misclass_words)
            common_words_by_pair[f"{main_class} - {misclass}"] = common_words
    return common_words_by_pair

def run_misclassification_extraction(input_data, threshold=0.2, epochs=3):
    if isinstance(input_data, str):
        df = pd.read_csv(input_data)
    else:
        df = input_data.copy()
    num_labels = df['class_label'].nunique()
    cm, label_encoder, df_out = train_model(df, num_labels=num_labels, epochs=epochs)
    mis_rates, _ = compute_misclassification_rates(cm, label_encoder)
    significant_pairs = extract_significant_pairs(mis_rates, threshold=threshold)
    common_words_by_pair = extract_common_words(df_out, significant_pairs)
    return common_words_by_pair