# Xiaa M1

Xiaa M1 is a 299M-parameter bilingual chat foundation model built from scratch by Xiaa AI.

## Chat with Xiaa M1

```bash
python scripts/chat.py
```

## Training Pipeline

```bash
# Step 1: download data + train tokenizer
python scripts/prepare_data.py

# Step 2: pre-train on 10B tokens
python scripts/train.py

# Step 3: SFT for chat
python scripts/finetune_chat.py

# Step 4: chat!
python scripts/chat.py
```

## Status Checklist

- [x] Base transformer architecture (`xiaa/model.py`)
- [x] Core model configuration (`xiaa/config.py`)
- [x] SentencePiece tokenizer wrapper (`xiaa/tokenizer.py`)
- [x] ChatML formatting utilities (`xiaa/chat_format.py`)
- [x] Data preparation pipeline (`scripts/prepare_data.py`)
- [x] Pretraining loop with checkpoint resume (`scripts/train.py`)
- [x] Chat SFT loop + Hugging Face export (`scripts/finetune_chat.py`)
- [x] Interactive CLI chat interface (`scripts/chat.py`)
- [x] Kaggle training notebook (`notebooks/kaggle_train.ipynb`)
- [x] Python dependencies (`requirements.txt`)
