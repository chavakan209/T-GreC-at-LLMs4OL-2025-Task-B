import json
import torch
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

class MedicalTermDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if predictions.shape[-1] > 1:
        predictions = np.argmax(predictions, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'f1_micro': f1_score(labels, predictions, average='micro')
    }

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [d["term"] for d in data]
    labels = [d["types"][0] if d["types"] else "unknown" for d in data]
    return texts, labels

def train_model(dataset_path, model_name, output_dir, label_encoder):
    texts, labels = load_data(dataset_path)
    encoded_labels = label_encoder.transform(labels)
    num_labels = len(label_encoder.classes_)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = MedicalTermDataset(X_train, y_train, tokenizer)
    val_dataset = MedicalTermDataset(X_val, y_val, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    if "deberta" in model_name.lower():
        target_modules = ["query_proj", "key_proj", "value_proj"]
    else:
        target_modules = ["query", "key", "value"]

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=200,
        eval_steps=200,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=50,
        learning_rate=5e-4,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=f"{output_dir}/logs",
        report_to=None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(f"{output_dir}/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

def main():
    dataset_model_pairs = [
        ("term_typing_train_data.json", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", "./PubMedBERT_ORG"),
        ("term_typing_augmented_data.json", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", "./PubMedBERT_AUG"),
        ("term_typing_augmented_data.json", "dmis-lab/biobert-base-cased-v1.1", "./BioBERT_AUG"),
        ("term_typing_augmented_data.json", "microsoft/deberta-v3-base", "./DeBERTa_AUG"),
        ("term_typing_augmented_data.json", "roberta-base", "./RoBERTa_AUG")
    ]

    all_labels = []
    for path, _, _ in dataset_model_pairs:
        _, labels = load_data(path)
        all_labels.extend(labels)

    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    for path, model_name, out_dir in dataset_model_pairs:
        print(f"\nTraining on {path} with {model_name} => saving to {out_dir}")
        train_model(path, model_name, out_dir, label_encoder)

if __name__ == "__main__":
    main()
