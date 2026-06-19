# -*- coding: utf-8 -*-
"""
BanglaBERT: NER (Multi-word BIO) + Sentiment Analysis
======================================================
This script fine-tunes BanglaBERT (csebuetnlp/banglabert) for:
  1. Named Entity Recognition (NER) - extracting product entities using BIO tagging
  2. Sentiment Analysis (SA) - classifying reviews as Negative / Neutral / Positive

Training data: Gemini 3.5-Flash Synthetic Dataset
Test data:     Test.csv
"""

# ============================
# A. Install Required Libraries
# ============================
# Run these in terminal before first use:
#   pip install datasets tokenizers seqeval transformers torch sentencepiece evaluate 'accelerate>=1.1.0'

# ============================
# B. Import Libraries
# ============================
import os
os.environ["WANDB_DISABLED"] = "true"

import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, auc
)
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForTokenClassification,
    AutoModelForSequenceClassification, TrainingArguments, Trainer,
    DataCollatorForTokenClassification, DataCollatorWithPadding
)
import evaluate

# ============================
# C. Load Datasets
# ============================
# Load the Gemini 3.5-Flash Synthetic Dataset (training) and Test.csv (evaluation).
# Both CSVs contain columns: id, review, sentiment, bio_tagged_review.
TRAIN_CSV = 'Synthetic Datasets/Gemini 3.5-Flash_Synthetic_Dataset.csv'
TEST_CSV  = 'Synthetic Datasets/Test.csv'

df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)

print(f"Training samples: {len(df_train)}")
print(f"Test samples:     {len(df_test)}")

# ============================
# D. Parse BIO Tags -> Tokens & NER Labels
# ============================
# Each bio_tagged_review string has tokens annotated as token-B or token-I.
# We parse them into parallel lists of tokens and BIO labels
# (B-Product, I-Product, O). For the test set, if bio_tagged_review
# is missing we fall back to whitespace tokenization.
def parse_bio_tags(bio_text):
    tokens, labels = [], []
    for part in bio_text.split():
        match = re.match(r'^(.*?)(?:[,\s।]*)-(B|I)$', part)
        if match:
            token, tag = match.group(1), match.group(2)
            labels.append('B-Product' if tag == 'B' else 'I-Product')
        else:
            token = part
            labels.append('O')
        tokens.append(token)
    return tokens, labels

df_train['tokens']   = df_train['bio_tagged_review'].apply(lambda x: parse_bio_tags(x)[0])
df_train['ner_tags'] = df_train['bio_tagged_review'].apply(lambda x: parse_bio_tags(x)[1])

if 'bio_tagged_review' in df_test.columns:
    df_test['tokens']   = df_test['bio_tagged_review'].apply(lambda x: parse_bio_tags(x)[0])
    df_test['ner_tags'] = df_test['bio_tagged_review'].apply(lambda x: parse_bio_tags(x)[1])
else:
    df_test['tokens'] = df_test['review'].apply(lambda x: x.split())

# ============================
# E. Label Mappings
# ============================
# Define bidirectional mappings between label strings and integer IDs.
# Three labels: O (non-entity), B-Product (begin), I-Product (inside).
label_list = ['O', 'B-Product', 'I-Product']
id2label   = {i: label for i, label in enumerate(label_list)}
label2id   = {label: i for i, label in enumerate(label_list)}
num_labels = len(label_list)

# ============================
# F. Split Training Data (80/20)
# ============================
# Create a stratified 80/20 train/validation split.
# Stratification on sentiment keeps class distribution consistent.
X = df_train[['id', 'review', 'tokens', 'ner_tags']]
y = df_train['sentiment']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)},  Validation size: {len(X_val)}")
X_train = X_train.reset_index(drop=True)
X_val   = X_val.reset_index(drop=True)

# ============================
# G. NER Tokenizer & Alignment (BanglaBERT)
# ============================
# Load the BanglaBERT tokenizer. Sub-word tokenization splits a single
# word into multiple tokens, so we align NER labels using word_ids():
# only the first sub-token of each word gets the true label; the rest
# get -100 (ignored during loss computation).
BANGLABERT = "csebuetnlp/banglabert"

tokenizer_ner = AutoTokenizer.from_pretrained(BANGLABERT)

def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        padding=False
    )
    all_labels = []
    for i, label_seq in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label_seq[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs['labels'] = all_labels
    return tokenized_inputs

def create_ner_dataset(df, tokenizer, label2id):
    tokenized = tokenize_and_align_labels(
        {'tokens': df['tokens'].tolist(), 'ner_tags': df['ner_tags'].tolist()},
        tokenizer, label2id
    )
    return Dataset.from_dict({
        'input_ids':      tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels':         tokenized['labels'],
        'tokens':         df['tokens'].tolist(),
        'ner_tags':       df['ner_tags'].tolist(),
        'id':             df['id'].tolist() if 'id' in df else list(range(len(df)))
    })

train_dataset_ner = create_ner_dataset(X_train, tokenizer_ner, label2id)
val_dataset_ner   = create_ner_dataset(X_val,   tokenizer_ner, label2id)

if 'tokens' in df_test.columns and 'ner_tags' in df_test.columns:
    test_dataset_ner = create_ner_dataset(df_test, tokenizer_ner, label2id)
else:
    test_dataset_ner = None

print(f"NER datasets created -- train: {len(train_dataset_ner)}, val: {len(val_dataset_ner)}")

# ============================
# H. Train Final NER Model (BanglaBERT Token Classification)
# ============================
# Merge train + val into one full training set, then fine-tune BanglaBERT
# for token classification (3 epochs). Evaluate on the held-out validation
# set using the seqeval metric and save the best checkpoint.
print("\n=== Training NER model on full training data ===")

full_train = pd.concat([X_train, X_val], ignore_index=True)
full_train['sentiment'] = pd.concat([y_train, y_val], ignore_index=True)
full_train_ds = create_ner_dataset(full_train, tokenizer_ner, label2id)

# Build model from BanglaBERT pretrained weights
config = AutoConfig.from_pretrained(BANGLABERT)
config.num_labels = num_labels
config.id2label   = id2label
config.label2id   = label2id
model_ner = AutoModelForTokenClassification.from_pretrained(
    BANGLABERT, config=config
)

args_ner = TrainingArguments(
    output_dir="ner_final",
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
)

data_collator_ner = DataCollatorForTokenClassification(tokenizer_ner)
metric_ner = evaluate.load("seqeval")

def compute_metrics_ner(eval_preds):
    pred_logits, labels = eval_preds
    pred_logits = np.argmax(pred_logits, axis=2)
    predictions = [
        [label_list[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(pred_logits, labels)
    ]
    references = [
        [label_list[l] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(pred_logits, labels)
    ]
    results = metric_ner.compute(predictions=predictions, references=references)
    return {
        "precision": results["overall_precision"],
        "recall":    results["overall_recall"],
        "f1":        results["overall_f1"],
        "accuracy":  results["overall_accuracy"],
    }

trainer_ner = Trainer(
    model=model_ner,
    args=args_ner,
    train_dataset=full_train_ds,
    eval_dataset=val_dataset_ner,
    data_collator=data_collator_ner,
    processing_class=tokenizer_ner,
    compute_metrics=compute_metrics_ner
)

trainer_ner.train()

# Evaluate on test set if available
if test_dataset_ner is not None:
    test_results_ner = trainer_ner.evaluate(test_dataset_ner)
    print("\n=== NER Test Results ===")
    print(test_results_ner)

model_ner.save_pretrained("ner_model_final")
tokenizer_ner.save_pretrained("ner_tokenizer")
print("\nNER model saved.")

# ============================
# I. Sentiment Analysis - Tokenize & Prepare Datasets (BanglaBERT)
# ============================
# Load a separate BanglaBERT tokenizer for sequence classification.
# Tokenize the raw review text and create Dataset objects for each split.
sa_tokenizer = AutoTokenizer.from_pretrained(BANGLABERT)

def tokenize_sa(reviews, tokenizer):
    return tokenizer(reviews, truncation=True, padding=False)

X_train_sa      = X_train['review'].tolist()
y_train_sa      = y_train.tolist()
X_val_sa        = X_val['review'].tolist()
y_val_sa        = y_val.tolist()
full_train_sa   = full_train['review'].tolist()
full_train_sa_y = full_train['sentiment'].tolist()

train_enc_sa      = tokenize_sa(X_train_sa,    sa_tokenizer)
val_enc_sa        = tokenize_sa(X_val_sa,      sa_tokenizer)
full_train_enc_sa = tokenize_sa(full_train_sa,  sa_tokenizer)

train_sa_dataset = Dataset.from_dict({
    'input_ids':      train_enc_sa['input_ids'],
    'attention_mask': train_enc_sa['attention_mask'],
    'labels':         y_train_sa,
})
val_sa_dataset = Dataset.from_dict({
    'input_ids':      val_enc_sa['input_ids'],
    'attention_mask': val_enc_sa['attention_mask'],
    'labels':         y_val_sa,
})
full_train_sa_dataset = Dataset.from_dict({
    'input_ids':      full_train_enc_sa['input_ids'],
    'attention_mask': full_train_enc_sa['attention_mask'],
    'labels':         full_train_sa_y,
})

print(f"SA datasets created -- train: {len(train_sa_dataset)}, val: {len(val_sa_dataset)}")

# ============================
# J. Train Sentiment Analysis Model (BanglaBERT Sequence Classification)
# ============================
# Fine-tune BanglaBERT for 3-class sentiment classification
# (Negative / Neutral / Positive) on the full training set.
# The best checkpoint (by validation accuracy) is kept.
print("\n=== Training final SA model on full training data ===")

model_sa = AutoModelForSequenceClassification.from_pretrained(
    BANGLABERT,
    num_labels=3,
    id2label={0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"},
    label2id={"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
)

args_sa = TrainingArguments(
    output_dir="sa_final",
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
)

data_collator_sa = DataCollatorWithPadding(sa_tokenizer)

def compute_metrics_sa_final(eval_pred):
    preds  = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    return {
        'precision': precision_score(labels, preds, average='weighted'),
        'recall':    recall_score(labels, preds, average='weighted'),
        'f1':        f1_score(labels, preds, average='weighted'),
        'accuracy':  accuracy_score(labels, preds)
    }

trainer_sa = Trainer(
    model=model_sa,
    args=args_sa,
    train_dataset=full_train_sa_dataset,
    eval_dataset=val_sa_dataset,
    processing_class=sa_tokenizer,
    data_collator=data_collator_sa,
    compute_metrics=compute_metrics_sa_final
)

trainer_sa.train()

# ============================
# K. Evaluate SA on Test Set
# ============================
# Tokenize the test-set reviews and run evaluation.
# If the test CSV lacks a sentiment column, dummy labels are used.
test_reviews  = df_test['review'].tolist()
test_enc_sa   = tokenize_sa(test_reviews, sa_tokenizer)

if 'sentiment' in df_test.columns:
    test_labels = df_test['sentiment'].tolist()
else:
    test_labels = [0] * len(df_test)
    print("Warning: No sentiment column in test set - using dummy labels.")

test_sa_dataset = Dataset.from_dict({
    'input_ids':      test_enc_sa['input_ids'],
    'attention_mask': test_enc_sa['attention_mask'],
    'labels':         test_labels
})

test_results_sa = trainer_sa.evaluate(test_sa_dataset)
print("\n=== SA Test Results ===")
print(test_results_sa)

model_sa.save_pretrained("sa_model_final")
sa_tokenizer.save_pretrained("sa_tokenizer")
print("SA model saved.")

# ============================
# L. Inference Example (with device fix)
# ============================
# Demonstrate end-to-end inference on a sample Bangla review:
# 1. NER pipeline - extract product entities from the text
# 2. Sentiment classification - predict overall sentiment
from transformers import pipeline

# NER pipeline
ner_pipe = pipeline(
    "ner",
    model="ner_model_final",
    tokenizer="ner_tokenizer",
    aggregation_strategy="simple"
)

sample_review = "ফেস ওয়াশ টা অনেক সুন্দর, অসংখ্য ধন্যবাদ ডেলিভারি খুব দ্রুত দেয়ার জন্য।"
print("\nSample review:", sample_review)
print("NER results:", ner_pipe(sample_review))

# Sentiment inference with device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_sa.to(device)

enc = sa_tokenizer([sample_review], padding=True, truncation=True, return_tensors="pt")
enc = {k: v.to(device) for k, v in enc.items()}

with torch.no_grad():
    logits = model_sa(**enc).logits
    pred = logits.argmax(dim=1).item()

sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
print(f"Sentiment: {sentiment_map[pred]}")

print("\nAll tasks completed successfully!")
