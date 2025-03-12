import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import DebertaTokenizer, Trainer, TrainingArguments, DebertaForSequenceClassification, TrainerCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datasets import Dataset
from torch.nn import CrossEntropyLoss
import seaborn as sns
import matplotlib.pyplot as plt

def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['class_label'],
        test_size=0.2, stratify=df['class_label']
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def map_labels(y_train, y_valid, y_test):
    all_labels = pd.concat([y_train, y_valid, y_test]).unique()
    label_mapping = {label: idx for idx, label in enumerate(all_labels)}
    y_train_numeric = [label_mapping[label] for label in y_train]
    y_valid_numeric = [label_mapping[label] for label in y_valid]
    y_test_numeric = [label_mapping[label] for label in y_test]
    return label_mapping, y_train_numeric, y_valid_numeric, y_test_numeric

def compute_weights(y_train_numeric, device):
    classes = np.unique(y_train_numeric)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_numeric)
    return torch.tensor(weights, dtype=torch.float).to(device)

def tokenize_texts(tokenizer, texts):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

def create_dataset(encodings, labels, device):
    return Dataset.from_dict({
        'input_ids': encodings['input_ids'].to(device),
        'attention_mask': encodings['attention_mask'].to(device),
        'labels': torch.tensor(labels, dtype=torch.long).to(device)
    })

class DebertaForWeightedClassification(DebertaForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = class_weights

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            **kwargs
        )
        logits = outputs.logits
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits, labels)
        return {"loss": loss, "logits": logits}

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        labels = inputs.pop("labels").to(device)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model(**inputs)
        logits = outputs['logits']
        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

class LogMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"Epoch {state.epoch:.1f}: {metrics}")

def plot_confusion_matrix(cm, labels, output_path):
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=labels, yticklabels=labels)
    # Customize the x-axis ticks and labels
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, ha='right')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def save_test_results(output_file, test_results, cm, y_test_numeric, y_pred):
    with open(output_file, "w") as f:
        f.write("Test Results:\n")
        for key, value in test_results.items():
            f.write(f"{key}: {value}\n")
        f.write("\nConfusion Matrix:\n")
        for row in cm:
            f.write(" ".join(map(str, row)) + "\n")
    print(f"Test results saved to {output_file}")

def run_classification(df):
    # Split data and map labels
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)
    label_mapping, y_train_numeric, y_valid_numeric, y_test_numeric = map_labels(y_train, y_valid, y_test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = compute_weights(y_train_numeric, device)
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    train_encodings = tokenize_texts(tokenizer, X_train)
    valid_encodings = tokenize_texts(tokenizer, X_valid)
    test_encodings = tokenize_texts(tokenizer, X_test)
    train_dataset = create_dataset(train_encodings, y_train_numeric, device)
    valid_dataset = create_dataset(valid_encodings, y_valid_numeric, device)
    test_dataset = create_dataset(test_encodings, y_test_numeric, device)
    model = DebertaForWeightedClassification.from_pretrained(
        'microsoft/deberta-base',
        num_labels=len(label_mapping),
        class_weights=class_weights
    ).to(device)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=20,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=8,
        eval_strategy='epoch',
        save_total_limit=2,
        logging_dir='./logs',
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=10,
        report_to="none"
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )
    saved_model_dir = './saved_model'
    if not os.path.exists(saved_model_dir):
        print("Starting training...")
        trainer.train()
        trainer.save_model(saved_model_dir)
    else:
        print("Loading saved model...")
        model = DebertaForSequenceClassification.from_pretrained(saved_model_dir).to(device)
    print("Evaluating the model on the test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"Test Results: {test_results}")
    test_preds = trainer.predict(test_dataset)
    y_pred = np.argmax(test_preds.predictions, axis=1)
    cm_final = confusion_matrix(y_test_numeric, y_pred)
    labels_sorted = [k for k, v in sorted(label_mapping.items(), key=lambda x: x[1])]
    plot_confusion_matrix(cm_final, labels_sorted, 'confusion_matrix.png')
    save_test_results("test_results.txt", test_results, cm_final, y_test_numeric, y_pred)