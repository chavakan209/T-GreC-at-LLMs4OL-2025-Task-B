# T-GreC-at-LLMs4OL-2025-Task-B# T-GreC: Term-Typing with LLM Embeddings + k-NN

This repository contains the code and experiments for our submission to **LLMs4OL 2025 Task B**, focusing on **term-typing** using large language models (LLMs) and **k-nearest neighbors (k-NN)**.

## 🧠 Project Summary

We evaluate four transformer-based models:
- `PubMedBERT`
- `BioBERT`
- `DeBERTa-v3-base`
- `RoBERTa-base`

Two methods are used:
1. **Fine-tuning with LoRA** for direct classification.
2. **Extracting embeddings** from LoRA-adapted models and using them in **k-NN**.

Tested on three datasets from the challenge:
- OBI (main focus)
- MatOnto
- SWEET
