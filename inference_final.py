import json
import torch
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)
from peft import PeftModel

BASE_MODELS = {
    "roberta": {
        "pretrained_name": "roberta-base",
    },
    "pubmed_org": {
        "pretrained_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    },
    "pubmed_aug": {
        "pretrained_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    },
    "biobert": {
        "pretrained_name": "dmis-lab/biobert-base-cased-v1.1",
    },
    "deberta": {
        "pretrained_name": "microsoft/deberta-v3-base",
    },
}

def load_main_model(model_path, model_key):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with open(f"{model_path}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    num_labels = len(label_encoder.classes_)
    base_info = BASE_MODELS[model_key]
    pretrained_name = base_info["pretrained_name"]
    base_model = AutoModelForSequenceClassification.from_pretrained(pretrained_name, num_labels=num_labels).to(device)
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    return model, tokenizer, label_encoder, device

def encode_with_lora_encoder(texts, tokenizer, encoder, device, batch_size=16, max_length=128):
    embeddings = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=max_length, return_tensors='pt').to(device)
            outputs = encoder.base_model(**encoded)
            cls_embed = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embed.cpu())
    return torch.cat(embeddings).numpy()

def knn_probabilities(X_embed, knn_model, y_train, k):
    distances, indices = knn_model.kneighbors(X_embed, n_neighbors=k)
    probs = []
    for idx_set in indices:
        votes = np.sum(y_train[idx_set], axis=0) / k
        probs.append(votes)
    return np.vstack(probs)

def predict_ensemble(
    term, main_model, tokenizer, encoder, device,
    knn_k1, y_train, label_encoder,
    alpha_main=0.75, alpha_k1=0.25,
    use_main=True, use_knn1=True,
    max_length=128
):
    combined_probs = 0
    total_weight = 0

    if use_main:
        encoded = tokenizer(term, truncation=True, padding='max_length',
                            max_length=max_length, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = main_model(**encoded).logits
            main_probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
        combined_probs += alpha_main * main_probs
        total_weight += alpha_main

    embed = None
    if use_knn1 and encoder is not None:
        embed = encode_with_lora_encoder([term], tokenizer, encoder, device)

    if use_knn1 and knn_k1 is not None:
        knn1_probs = knn_probabilities(embed, knn_k1, y_train, k=1)[0]
        combined_probs += alpha_k1 * knn1_probs
        total_weight += alpha_k1

    combined_probs /= total_weight
    top1_idx = np.argmax(combined_probs)
    top1_label = label_encoder.inverse_transform([top1_idx])[0]
    return top1_label


def run_experiment_for_model(
    model_key,
    model_path,
    train_file_path="term_typing_train_data.json",
    test_file_path="obi_term_typing_test_data.json",
    output_file_path="predictions.json",
    alpha_main=0.75,
    alpha_k1=0.25,
    use_main=True,
    use_knn1=True
):
    print(f"Loading model ({model_key}) from {model_path} ...")
    model, tokenizer, label_encoder, device = load_main_model(model_path, model_key)
    encoder = model.base_model if use_knn1 else None

    knn_k1 = y_train = None
    if use_knn1:
        with open(train_file_path, "r") as f:
            train_data = json.load(f)
        train_terms = [item["term"] for item in train_data]
        train_types = [item["types"] for item in train_data]
        mlb = MultiLabelBinarizer()
        y_train = mlb.fit_transform(train_types)
        train_embed = encode_with_lora_encoder(train_terms, tokenizer, encoder, device)
        knn_k1 = NearestNeighbors(n_neighbors=1, metric="cosine").fit(train_embed)

    with open(test_file_path, "r") as f:
        test_data = json.load(f)

    print(f"Running inference on {len(test_data)} samples...")
    predictions = []
    for i, sample in enumerate(test_data):
        term_id = sample["id"]
        term = sample["term"]

        pred_type = predict_ensemble(
            term, model, tokenizer, encoder, device,
            knn_k1, y_train, label_encoder,
            alpha_main=alpha_main, alpha_k1=alpha_k1,
            use_main=use_main, use_knn1=use_knn1
        )

        predictions.append({
            "id": term_id,
            "types": [pred_type]
        })

        if i % 100 == 0:
            print(f"Processed {i}/{len(test_data)}")

    print(f"Saving predictions to {output_file_path} ...")
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print("Inference completed!")

if __name__ == "__main__":
    MODELS = {
        "roberta": "./RoBERTa_AUG",
        "pubmed_org": "./PubMedBERT_ORG",
        "pubmed_aug": "./PubMedBERT_AUG",
        "biobert": "./BioBERT_AUG",
        "deberta": "./DeBERTa_AUG",
    }
    TRAIN_FILE = "term_typing_train_data.json"
    TEST_FILE = "obi_term_typing_test_data.json"
    for model_name, model_path in MODELS.items():
        run_experiment_for_model(
            model_key=model_name,
            model_path=model_path,
            train_file_path=TRAIN_FILE,
            test_file_path=TEST_FILE,
            output_file_path=f"final_{model_name}_main.json",
            alpha_main=1.0,
            alpha_k1=0.0,
            use_main=True,
            use_knn1=False
        )
        run_experiment_for_model(
            model_key=model_name,
            model_path=model_path,
            train_file_path=TRAIN_FILE,
            test_file_path=TEST_FILE,
            output_file_path=f"final_{model_name}_knn.json",
            alpha_main=0.0,
            alpha_k1=1.0,
            use_main=False,
            use_knn1=True
        )

    # RoBERTa MatOnto
    MODELS = {
        ("roberta","./model_roberta_MatORG"),
        ("roberta","./model_roberta_MatAug"),
    }
    TRAIN_FILE = "term_typing_train_data_Mat.json"
    TEST_FILE = "matonto_term_typing_test_data.json"
    for (model_name, model_path) in MODELS.items():
        run_experiment_for_model(
            model_key=model_name,
            model_path=model_path,
            train_file_path=TRAIN_FILE,
            test_file_path=TEST_FILE,
            output_file_path=f"final_{model_path[-6:]}_main.json",
            alpha_main=1.0,
            alpha_k1=0.0,
            use_main=True,
            use_knn1=False
        )
        run_experiment_for_model(
            model_key=model_name,
            model_path=model_path,
            train_file_path=TRAIN_FILE,
            test_file_path=TEST_FILE,
            output_file_path=f"final_{model_path[-6:]}_knn.json",
            alpha_main=0.0,
            alpha_k1=1.0,
            use_main=False,
            use_knn1=True
        )


    # RoBERTa SWEET
    MODELS = [
        ("roberta","./model_roberta_SWEETORG"),
        ("roberta","./model_roberta_SWEETAug"),
    ]
    TRAIN_FILE = "term_typing_train_data_sweet.json"
    TEST_FILE = "sweet_term_typing_test_data.json"
    for (model_name, model_path) in MODELS:
        run_experiment_for_model(
            model_key=model_name,
            model_path=model_path,
            train_file_path=TRAIN_FILE,
            test_file_path=TEST_FILE,
            output_file_path=f"final_{model_path[-8:]}_main.json",
            alpha_main=1.0,
            alpha_k1=0.0,
            use_main=True,
            use_knn1=False
        )

        run_experiment_for_model(
            model_key=model_name,
            model_path=model_path,
            train_file_path=TRAIN_FILE,
            test_file_path=TEST_FILE,
            output_file_path=f"final_{model_path[-8:]}_knn.json",
            alpha_main=0.0,
            alpha_k1=1.0,
            use_main=False,
            use_knn1=True
        )