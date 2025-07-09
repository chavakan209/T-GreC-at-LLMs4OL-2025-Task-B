import json
import random
import uuid
from tqdm import tqdm

INPUT_JSON = "term_typing_train_data.json"
OUTPUT_JSON = "term_typing_augmented_data.json"
AUG_PER_TERM = 3

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    original_data = json.load(f)

augmented_data = []

def random_insert(text):
    if not text:
        return text
    idx = random.randint(0, len(text)-1)
    char = random.choice("abcdefghijklmnopqrstuvwxyz")
    return text[:idx] + char + text[idx:]

def random_delete(text):
    if len(text) <= 1:
        return text
    idx = random.randint(0, len(text)-1)
    return text[:idx] + text[idx+1:]

def random_swap(text):
    if len(text) < 2:
        return text
    idx = random.randint(0, len(text)-2)
    return text[:idx] + text[idx+1] + text[idx] + text[idx+2:]

def random_substitute(text):
    if not text:
        return text
    idx = random.randint(0, len(text)-1)
    char = random.choice("abcdefghijklmnopqrstuvwxyz")
    return text[:idx] + char + text[idx+1:]

def augment_term(term):
    ops = [random_insert, random_delete, random_swap, random_substitute]
    return random.choice(ops)(term)

for item in tqdm(original_data):
    term = item["term"]
    types = item["types"]
    augmented_data.append(item)
    for _ in range(AUG_PER_TERM):
        new_term = augment_term(term)
        if new_term.strip() != term.strip():
            augmented_data.append({
                "id": str(uuid.uuid4()),
                "term": new_term,
                "types": types
            })
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(augmented_data, f, indent=2, ensure_ascii=False)

print(f"\nSaved to {OUTPUT_JSON}")
