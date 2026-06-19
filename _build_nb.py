#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_build_nb.py
Generates 3 Jupyter notebooks in Codes/ for BanglaBERT, mBERT, and XLM-RoBERTa.
Each notebook: NER (BIO) + Sentiment Analysis + Zero-Shot Classification + Report Saving.
"""
import json, os

# ── notebook cell helpers ──────────────────────────────────────────────
def _md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def _code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

# ── parameterised notebook builder ─────────────────────────────────────
def build_notebook(model_name, model_id, short, display):
    """Return a notebook dict for the given model configuration.

    Parameters
    ----------
    model_name : str   – Python variable name  (e.g. "BANGLABERT")
    model_id   : str   – HuggingFace identifier (e.g. "csebuetnlp/banglabert")
    short      : str   – short tag for file names (e.g. "banglabert")
    display    : str   – human-readable name      (e.g. "BanglaBERT")
    """
    cells = []
    md   = lambda s: cells.append(_md(s))
    code = lambda s: cells.append(_code(s))

    # ── Title ──────────────────────────────────────────────────────────
    md([
        f"# {display}: NER (Multi-word BIO) + Sentiment Analysis + Zero-Shot Classification\n",
        "\n",
        f"This notebook fine-tunes **{display}** (`{model_id}`) for:\n",
        "1. **Named Entity Recognition (NER)** – extracting product entities using BIO tagging\n",
        "2. **Sentiment Analysis (SA)** – classifying reviews as Negative / Neutral / Positive\n",
        "3. **Zero-Shot Classification** – classifying reviews without task-specific training\n",
        "\n",
        "**Training data:** Gemini 3.5-Flash Synthetic Dataset  \n",
        "**Test data:** Test.csv  \n",
        f"**Model:** `{model_id}`"
    ])

    # ── A. Install ─────────────────────────────────────────────────────
    md([
        "## A. Install Required Libraries\n",
        "Install Hugging Face `transformers`, `datasets`, `evaluate`, `seqeval` (for token-level NER metrics),\n",
        "`torch` as the deep-learning backend, `sentencepiece` for sub-word tokenization,\n",
        "and `accelerate>=1.1.0` (required by `Trainer` in newer transformers)."
    ])
    code([
        "# -*- coding: utf-8 -*-\n",
        "!pip install datasets tokenizers seqeval -q\n",
        "!pip install transformers\n",
        "!pip install torch\n",
        "!pip install sentencepiece\n",
        "!pip install evaluate\n",
        "!pip install 'accelerate>=1.1.0' -q"
    ])

    # ── B. Import ──────────────────────────────────────────────────────
    md([
        "## B. Import Libraries\n",
        "Import standard data-science and NLP libraries. We disable Weights & Biases logging\n",
        "via the `WANDB_DISABLED` environment variable so training runs stay local."
    ])
    code([
        "import os\n",
        "os.environ[\"WANDB_DISABLED\"] = \"true\"\n",
        "\n",
        "import re\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import torch\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import (\n",
        "    precision_score, recall_score, f1_score, accuracy_score,\n",
        "    confusion_matrix, classification_report\n",
        ")\n",
        "from datasets import Dataset\n",
        "from transformers import (\n",
        "    AutoTokenizer, AutoConfig, AutoModelForTokenClassification,\n",
        "    AutoModelForSequenceClassification, TrainingArguments, Trainer,\n",
        "    DataCollatorForTokenClassification, DataCollatorWithPadding,\n",
        "    pipeline\n",
        ")\n",
        "import evaluate"
    ])

    # ── C. Load Data ───────────────────────────────────────────────────
    md([
        "## C. Load Datasets\n",
        "Load the **Gemini 3.5-Flash Synthetic Dataset** as the training set and the **Test** dataset\n",
        "from the `Synthetic Datasets/` folder. Both CSVs contain columns: `id`, `review`, `sentiment`,\n",
        "and `bio_tagged_review`."
    ])
    code([
        "TRAIN_CSV = 'Synthetic Datasets/Gemini 3.5-Flash_Synthetic_Dataset.csv'\n",
        "TEST_CSV  = 'Synthetic Datasets/Test.csv'\n",
        "\n",
        "df_train = pd.read_csv(TRAIN_CSV)\n",
        "df_test  = pd.read_csv(TEST_CSV)\n",
        "\n",
        "print(f\"Training samples: {len(df_train)}\")\n",
        "print(f\"Test samples:     {len(df_test)}\")\n",
        "df_train.head()"
    ])

    # ── D. Parse BIO ──────────────────────────────────────────────────
    md([
        "## D. Parse BIO Tags into Tokens & NER Labels\n",
        "Each `bio_tagged_review` string contains tokens annotated as `token-B` or `token-I`.\n",
        "We parse them into parallel lists of **tokens** and **BIO labels** (`B-Product`, `I-Product`, `O`).\n",
        "For the test set, if `bio_tagged_review` is missing we fall back to whitespace tokenization."
    ])
    code([
        "def parse_bio_tags(bio_text):\n",
        "    tokens, labels = [], []\n",
        "    for part in bio_text.split():\n",
        "        match = re.match(r'^(.*?)(?:[,\\s\u0964]*)-(B|I)$', part)\n",
        "        if match:\n",
        "            token, tag = match.group(1), match.group(2)\n",
        "            labels.append('B-Product' if tag == 'B' else 'I-Product')\n",
        "        else:\n",
        "            token = part\n",
        "            labels.append('O')\n",
        "        tokens.append(token)\n",
        "    return tokens, labels\n",
        "\n",
        "# Parse training set\n",
        "df_train['tokens']   = df_train['bio_tagged_review'].apply(lambda x: parse_bio_tags(x)[0])\n",
        "df_train['ner_tags'] = df_train['bio_tagged_review'].apply(lambda x: parse_bio_tags(x)[1])\n",
        "\n",
        "# Parse test set (fall back to plain split if no BIO column)\n",
        "if 'bio_tagged_review' in df_test.columns:\n",
        "    df_test['tokens']   = df_test['bio_tagged_review'].apply(lambda x: parse_bio_tags(x)[0])\n",
        "    df_test['ner_tags'] = df_test['bio_tagged_review'].apply(lambda x: parse_bio_tags(x)[1])\n",
        "else:\n",
        "    df_test['tokens'] = df_test['review'].apply(lambda x: x.split())\n",
        "\n",
        "print(df_train[['review', 'tokens', 'ner_tags']].iloc[0])"
    ])

    # ── E. Label Mappings ─────────────────────────────────────────────
    md([
        "## E. Define Label Mappings\n",
        "Create bidirectional mappings between label strings and integer IDs.\n",
        "Our NER scheme has three labels: `O` (non-entity), `B-Product` (beginning of product),\n",
        "and `I-Product` (inside product)."
    ])
    code([
        "label_list = ['O', 'B-Product', 'I-Product']\n",
        "id2label   = {i: label for i, label in enumerate(label_list)}\n",
        "label2id   = {label: i for i, label in enumerate(label_list)}\n",
        "num_labels = len(label_list)\n",
        "\n",
        "print(f\"Labels ({num_labels}): {label_list}\")\n",
        "print(f\"label2id: {label2id}\")"
    ])

    # ── F. Split ──────────────────────────────────────────────────────
    md([
        "## F. Split Training Data (80/20)\n",
        "Create a stratified 80/20 train/validation split so we can monitor overfitting during training.\n",
        "Stratification on `sentiment` ensures the class distribution stays consistent across splits."
    ])
    code([
        "X = df_train[['id', 'review', 'tokens', 'ner_tags']]\n",
        "y = df_train['sentiment']\n",
        "\n",
        "X_train, X_val, y_train, y_val = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=42, stratify=y\n",
        ")\n",
        "X_train = X_train.reset_index(drop=True)\n",
        "X_val   = X_val.reset_index(drop=True)\n",
        "\n",
        "print(f\"Train size: {len(X_train)},  Validation size: {len(X_val)}\")"
    ])

    # ── G. NER Tokenizer ──────────────────────────────────────────────
    md([
        f"## G. NER Tokenizer & Label Alignment ({display})\n",
        f"Load the **{display}** tokenizer (`{model_id}`). Because sub-word tokenization\n",
        "splits a single word into multiple tokens, we align NER labels using `word_ids()`:\n",
        "only the **first sub-token** of each word receives the true label; the rest get `-100`\n",
        "(ignored during loss computation)."
    ])
    code([
        f"{model_name} = \"{model_id}\"\n",
        "\n",
        f"tokenizer_ner = AutoTokenizer.from_pretrained({model_name})\n",
        "\n",
        "def tokenize_and_align_labels(examples, tokenizer, label2id):\n",
        "    tokenized_inputs = tokenizer(\n",
        "        examples['tokens'],\n",
        "        truncation=True,\n",
        "        is_split_into_words=True,\n",
        "        padding=False\n",
        "    )\n",
        "    all_labels = []\n",
        "    for i, label_seq in enumerate(examples['ner_tags']):\n",
        "        word_ids = tokenized_inputs.word_ids(batch_index=i)\n",
        "        previous_word_idx = None\n",
        "        label_ids = []\n",
        "        for word_idx in word_ids:\n",
        "            if word_idx is None:\n",
        "                label_ids.append(-100)\n",
        "            elif word_idx != previous_word_idx:\n",
        "                label_ids.append(label2id[label_seq[word_idx]])\n",
        "            else:\n",
        "                label_ids.append(-100)\n",
        "            previous_word_idx = word_idx\n",
        "        all_labels.append(label_ids)\n",
        "    tokenized_inputs['labels'] = all_labels\n",
        "    return tokenized_inputs\n",
        "\n",
        "def create_ner_dataset(df, tokenizer, label2id):\n",
        "    tokenized = tokenize_and_align_labels(\n",
        "        {'tokens': df['tokens'].tolist(), 'ner_tags': df['ner_tags'].tolist()},\n",
        "        tokenizer, label2id\n",
        "    )\n",
        "    return Dataset.from_dict({\n",
        "        'input_ids':      tokenized['input_ids'],\n",
        "        'attention_mask': tokenized['attention_mask'],\n",
        "        'labels':         tokenized['labels'],\n",
        "        'tokens':         df['tokens'].tolist(),\n",
        "        'ner_tags':       df['ner_tags'].tolist(),\n",
        "        'id':             df['id'].tolist() if 'id' in df else list(range(len(df)))\n",
        "    })\n",
        "\n",
        "train_dataset_ner = create_ner_dataset(X_train, tokenizer_ner, label2id)\n",
        "val_dataset_ner   = create_ner_dataset(X_val,   tokenizer_ner, label2id)\n",
        "\n",
        "if 'tokens' in df_test.columns and 'ner_tags' in df_test.columns:\n",
        "    test_dataset_ner = create_ner_dataset(df_test, tokenizer_ner, label2id)\n",
        "else:\n",
        "    test_dataset_ner = None\n",
        "\n",
        "print(f\"NER datasets created -- train: {len(train_dataset_ner)}, val: {len(val_dataset_ner)}\")"
    ])

    # ── H. Train NER ──────────────────────────────────────────────────
    md([
        f"## H. Train the NER Model ({display} Token Classification)\n",
        "Combine the train + validation splits into one full training set, then fine-tune\n",
        f"**{display}** for token classification (3 epochs). We evaluate on the held-out\n",
        "validation set using the `seqeval` metric and save the best checkpoint."
    ])
    code([
        "print(\"=== Training NER model on full training data ===\\n\")\n",
        "\n",
        "# Merge train + val for final training\n",
        "full_train = pd.concat([X_train, X_val], ignore_index=True)\n",
        "full_train['sentiment'] = pd.concat([y_train, y_val], ignore_index=True)\n",
        "full_train_ds = create_ner_dataset(full_train, tokenizer_ner, label2id)\n",
        "\n",
        f"# Build model from {display} pretrained weights\n",
        f"config = AutoConfig.from_pretrained({model_name})\n",
        "config.num_labels = num_labels\n",
        "config.id2label   = id2label\n",
        "config.label2id   = label2id\n",
        "model_ner = AutoModelForTokenClassification.from_pretrained(\n",
        f"    {model_name}, config=config\n",
        ")\n",
        "\n",
        f"args_ner = TrainingArguments(\n",
        f"    output_dir=\"ner_{short}_final\",\n",
        "    eval_strategy=\"epoch\",\n",
        "    learning_rate=3e-5,\n",
        "    per_device_train_batch_size=16,\n",
        "    per_device_eval_batch_size=16,\n",
        "    num_train_epochs=3,\n",
        "    weight_decay=0.01,\n",
        "    logging_steps=50,\n",
        "    save_strategy=\"epoch\",\n",
        "    load_best_model_at_end=True,\n",
        "    metric_for_best_model=\"eval_f1\",\n",
        ")\n",
        "\n",
        "data_collator_ner = DataCollatorForTokenClassification(tokenizer_ner)\n",
        "metric_ner = evaluate.load(\"seqeval\")\n",
        "\n",
        "def compute_metrics_ner(eval_preds):\n",
        "    pred_logits, labels = eval_preds\n",
        "    pred_logits = np.argmax(pred_logits, axis=2)\n",
        "    predictions = [\n",
        "        [label_list[p] for (p, l) in zip(pred, lab) if l != -100]\n",
        "        for pred, lab in zip(pred_logits, labels)\n",
        "    ]\n",
        "    references = [\n",
        "        [label_list[l] for (p, l) in zip(pred, lab) if l != -100]\n",
        "        for pred, lab in zip(pred_logits, labels)\n",
        "    ]\n",
        "    results = metric_ner.compute(predictions=predictions, references=references)\n",
        "    return {\n",
        "        \"precision\": results[\"overall_precision\"],\n",
        "        \"recall\":    results[\"overall_recall\"],\n",
        "        \"f1\":        results[\"overall_f1\"],\n",
        "        \"accuracy\":  results[\"overall_accuracy\"],\n",
        "    }\n",
        "\n",
        "trainer_ner = Trainer(\n",
        "    model=model_ner,\n",
        "    args=args_ner,\n",
        "    train_dataset=full_train_ds,\n",
        "    eval_dataset=val_dataset_ner,\n",
        "    data_collator=data_collator_ner,\n",
        "    processing_class=tokenizer_ner,\n",
        "    compute_metrics=compute_metrics_ner\n",
        ")\n",
        "\n",
        "trainer_ner.train()\n",
        "\n",
        "# Evaluate on test set if available\n",
        "if test_dataset_ner is not None:\n",
        "    test_results_ner = trainer_ner.evaluate(test_dataset_ner)\n",
        "    print(\"\\n=== NER Test Results ===\")\n",
        "    print(test_results_ner)\n",
        "\n",
        f"model_ner.save_pretrained(\"ner_{short}_model_final\")\n",
        f"tokenizer_ner.save_pretrained(\"ner_{short}_tokenizer\")\n",
        "print(\"\\nNER model saved.\")"
    ])

    # ── I. SA Prep ────────────────────────────────────────────────────
    md([
        f"## I. Sentiment Analysis – Tokenize & Prepare Datasets ({display})\n",
        f"Load a separate **{display}** tokenizer for sequence classification.\n",
        "Tokenize the raw review text and create `Dataset` objects for train, validation, and test splits."
    ])
    code([
        f"sa_tokenizer = AutoTokenizer.from_pretrained({model_name})\n",
        "\n",
        "def tokenize_sa(reviews, tokenizer):\n",
        "    return tokenizer(reviews, truncation=True, padding=False)\n",
        "\n",
        "# Prepare text & labels for each split\n",
        "X_train_sa      = X_train['review'].tolist()\n",
        "y_train_sa      = y_train.tolist()\n",
        "X_val_sa        = X_val['review'].tolist()\n",
        "y_val_sa        = y_val.tolist()\n",
        "full_train_sa   = full_train['review'].tolist()\n",
        "full_train_sa_y = full_train['sentiment'].tolist()\n",
        "\n",
        "# Tokenize\n",
        "train_enc_sa      = tokenize_sa(X_train_sa,    sa_tokenizer)\n",
        "val_enc_sa        = tokenize_sa(X_val_sa,      sa_tokenizer)\n",
        "full_train_enc_sa = tokenize_sa(full_train_sa,  sa_tokenizer)\n",
        "\n",
        "# Build HuggingFace Datasets\n",
        "train_sa_dataset = Dataset.from_dict({\n",
        "    'input_ids':      train_enc_sa['input_ids'],\n",
        "    'attention_mask': train_enc_sa['attention_mask'],\n",
        "    'labels':         y_train_sa,\n",
        "})\n",
        "val_sa_dataset = Dataset.from_dict({\n",
        "    'input_ids':      val_enc_sa['input_ids'],\n",
        "    'attention_mask': val_enc_sa['attention_mask'],\n",
        "    'labels':         y_val_sa,\n",
        "})\n",
        "full_train_sa_dataset = Dataset.from_dict({\n",
        "    'input_ids':      full_train_enc_sa['input_ids'],\n",
        "    'attention_mask': full_train_enc_sa['attention_mask'],\n",
        "    'labels':         full_train_sa_y,\n",
        "})\n",
        "\n",
        "print(f\"SA datasets created -- train: {len(train_sa_dataset)}, val: {len(val_sa_dataset)}\")"
    ])

    # ── J. Train SA ───────────────────────────────────────────────────
    md([
        f"## J. Train the Sentiment Analysis Model ({display} Sequence Classification)\n",
        f"Fine-tune **{display}** for 3-class sentiment classification (Negative / Neutral / Positive)\n",
        "on the full training set. The best checkpoint (by validation accuracy) is kept."
    ])
    code([
        "print(\"=== Training final Sentiment Analysis model ===\\n\")\n",
        "\n",
        "model_sa = AutoModelForSequenceClassification.from_pretrained(\n",
        f"    {model_name},\n",
        "    num_labels=3,\n",
        "    id2label={0: \"NEGATIVE\", 1: \"NEUTRAL\", 2: \"POSITIVE\"},\n",
        "    label2id={\"NEGATIVE\": 0, \"NEUTRAL\": 1, \"POSITIVE\": 2}\n",
        ")\n",
        "\n",
        f"args_sa = TrainingArguments(\n",
        f"    output_dir=\"sa_{short}_final\",\n",
        "    eval_strategy=\"epoch\",\n",
        "    learning_rate=3e-5,\n",
        "    per_device_train_batch_size=16,\n",
        "    per_device_eval_batch_size=16,\n",
        "    num_train_epochs=3,\n",
        "    weight_decay=0.01,\n",
        "    logging_steps=50,\n",
        "    save_strategy=\"epoch\",\n",
        "    load_best_model_at_end=True,\n",
        "    metric_for_best_model=\"eval_accuracy\",\n",
        ")\n",
        "\n",
        "data_collator_sa = DataCollatorWithPadding(sa_tokenizer)\n",
        "\n",
        "def compute_metrics_sa(eval_pred):\n",
        "    preds  = eval_pred.predictions.argmax(-1)\n",
        "    labels = eval_pred.label_ids\n",
        "    return {\n",
        "        'precision': precision_score(labels, preds, average='weighted'),\n",
        "        'recall':    recall_score(labels, preds, average='weighted'),\n",
        "        'f1':        f1_score(labels, preds, average='weighted'),\n",
        "        'accuracy':  accuracy_score(labels, preds)\n",
        "    }\n",
        "\n",
        "trainer_sa = Trainer(\n",
        "    model=model_sa,\n",
        "    args=args_sa,\n",
        "    train_dataset=full_train_sa_dataset,\n",
        "    eval_dataset=val_sa_dataset,\n",
        "    processing_class=sa_tokenizer,\n",
        "    data_collator=data_collator_sa,\n",
        "    compute_metrics=compute_metrics_sa\n",
        ")\n",
        "\n",
        "trainer_sa.train()\n",
        "print(\"\\nSA model training complete.\")"
    ])

    # ── K. Eval SA Test ───────────────────────────────────────────────
    md([
        "## K. Evaluate Sentiment Model on Test Set\n",
        "Tokenize the test-set reviews and run evaluation. If the test CSV lacks a `sentiment`\n",
        "column, dummy labels are used (accuracy will be meaningless but the pipeline won't crash)."
    ])
    code([
        "test_reviews  = df_test['review'].tolist()\n",
        "test_enc_sa   = tokenize_sa(test_reviews, sa_tokenizer)\n",
        "\n",
        "if 'sentiment' in df_test.columns:\n",
        "    test_labels = df_test['sentiment'].tolist()\n",
        "else:\n",
        "    test_labels = [0] * len(df_test)\n",
        "    print(\"Warning: No sentiment column in test set - using dummy labels.\")\n",
        "\n",
        "test_sa_dataset = Dataset.from_dict({\n",
        "    'input_ids':      test_enc_sa['input_ids'],\n",
        "    'attention_mask': test_enc_sa['attention_mask'],\n",
        "    'labels':         test_labels\n",
        "})\n",
        "\n",
        "test_results_sa = trainer_sa.evaluate(test_sa_dataset)\n",
        "print(\"\\n=== SA Test Results ===\")\n",
        "print(test_results_sa)\n",
        "\n",
        f"model_sa.save_pretrained(\"sa_{short}_model_final\")\n",
        f"sa_tokenizer.save_pretrained(\"sa_{short}_tokenizer\")\n",
        "print(\"SA model saved.\")"
    ])

    # ── L. Inference ──────────────────────────────────────────────────
    md([
        "## L. Inference Example\n",
        "Demonstrate end-to-end inference on a sample Bangla review:\n",
        "1. **NER pipeline** – extract product entities from the text.\n",
        "2. **Sentiment classification** – predict the overall sentiment (Negative / Neutral / Positive)."
    ])
    code([
        "# --- NER inference ---\n",
        "ner_pipe = pipeline(\n",
        "    \"ner\",\n",
        f"    model=\"ner_{short}_model_final\",\n",
        f"    tokenizer=\"ner_{short}_tokenizer\",\n",
        "    aggregation_strategy=\"simple\"\n",
        ")\n",
        "\n",
        "sample_review = \"\u09ab\u09c7\u09b8 \u0993\u09af\u09bc\u09be\u09b6 \u099f\u09be \u0985\u09a8\u09c7\u0995 \u09b8\u09c1\u09a8\u09cd\u09a6\u09b0, \u0985\u09b8\u0982\u0996\u09cd\u09af \u09a7\u09a8\u09cd\u09af\u09ac\u09be\u09a6 \u09a1\u09c7\u09b2\u09bf\u09ad\u09be\u09b0\u09bf \u0996\u09c1\u09ac \u09a6\u09cd\u09b0\u09c1\u09a4 \u09a6\u09c7\u09af\u09bc\u09be\u09b0 \u099c\u09a8\u09cd\u09af\u0964\"\n",
        "print(\"Sample review:\", sample_review)\n",
        "print(\"NER results:\", ner_pipe(sample_review))\n",
        "\n",
        "# --- Sentiment inference ---\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "model_sa.to(device)\n",
        "\n",
        "enc = sa_tokenizer([sample_review], padding=True, truncation=True, return_tensors=\"pt\")\n",
        "enc = {k: v.to(device) for k, v in enc.items()}\n",
        "\n",
        "with torch.no_grad():\n",
        "    logits = model_sa(**enc).logits\n",
        "    pred = logits.argmax(dim=1).item()\n",
        "\n",
        "sentiment_map = {0: \"Negative\", 1: \"Neutral\", 2: \"Positive\"}\n",
        "print(f\"\\nPredicted sentiment: {sentiment_map[pred]}\")"
    ])

    # ── M. Zero-Shot Classification ───────────────────────────────────
    md([
        "## M. Zero-Shot Sentiment Classification\n",
        "Run **zero-shot classification** using a multilingual NLI model\n",
        "(`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual`) to classify Bangla reviews\n",
        "into Negative / Neutral / Positive **without any task-specific fine-tuning**.\n",
        "This provides a baseline comparison against the fine-tuned models."
    ])
    code([
        "# Load zero-shot pipeline (multilingual NLI-based)\n",
        "zs_model_name = \"MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual\"\n",
        "print(f\"Loading zero-shot model: {zs_model_name}\\n\")\n",
        "\n",
        "zs_pipe = pipeline(\n",
        "    \"zero-shot-classification\",\n",
        "    model=zs_model_name,\n",
        "    tokenizer=zs_model_name,\n",
        ")\n",
        "\n",
        "# Candidate labels for sentiment\n",
        "candidate_labels = [\"negative\", \"neutral\", \"positive\"]\n",
        "\n",
        "# Run on test set\n",
        "zs_results = []\n",
        "for idx, review in enumerate(df_test['review'].tolist()):\n",
        "    result = zs_pipe(review, candidate_labels=candidate_labels)\n",
        "    predicted_label = result['labels'][0]\n",
        "    predicted_score = result['scores'][0]\n",
        "    zs_results.append({\n",
        "        'id': idx + 1,\n",
        "        'review': review[:80],\n",
        "        'predicted_sentiment': predicted_label,\n",
        "        'confidence': round(predicted_score, 4),\n",
        "        'all_labels': result['labels'],\n",
        "        'all_scores': [round(s, 4) for s in result['scores']]\n",
        "    })\n",
        "    if (idx + 1) % 10 == 0:\n",
        "        print(f\"  Processed {idx + 1}/{len(df_test)} reviews\")\n",
        "\n",
        "df_zs = pd.DataFrame(zs_results)\n",
        "print(f\"\\nZero-shot classification complete for {len(df_zs)} reviews.\")\n",
        "df_zs.head(10)"
    ])

    # ── M.1 Zero-Shot Metrics ─────────────────────────────────────────
    md([
        "### M.1 Zero-Shot Evaluation Metrics\n",
        "Compare zero-shot predictions against ground-truth labels (if available)\n",
        "and compute standard classification metrics."
    ])
    code([
        "# Map zero-shot labels to integer IDs for comparison\n",
        "zs_label_map = {\"negative\": 0, \"neutral\": 1, \"positive\": 2}\n",
        "df_zs['pred_id'] = df_zs['predicted_sentiment'].map(zs_label_map)\n",
        "\n",
        "if 'sentiment' in df_test.columns:\n",
        "    true_labels = df_test['sentiment'].tolist()\n",
        "    zs_preds    = df_zs['pred_id'].tolist()\n",
        "\n",
        "    zs_accuracy  = accuracy_score(true_labels, zs_preds)\n",
        "    zs_precision = precision_score(true_labels, zs_preds, average='weighted')\n",
        "    zs_recall    = recall_score(true_labels, zs_preds, average='weighted')\n",
        "    zs_f1        = f1_score(true_labels, zs_preds, average='weighted')\n",
        "\n",
        "    print(\"=== Zero-Shot Classification Metrics ===\")\n",
        "    print(f\"  Accuracy:  {zs_accuracy:.4f}\")\n",
        "    print(f\"  Precision: {zs_precision:.4f}\")\n",
        "    print(f\"  Recall:    {zs_recall:.4f}\")\n",
        "    print(f\"  F1 Score:  {zs_f1:.4f}\")\n",
        "    print(\"\\nClassification Report:\")\n",
        "    print(classification_report(true_labels, zs_preds,\n",
        "          target_names=['Negative', 'Neutral', 'Positive']))\n",
        "\n",
        "    # Confusion matrix\n",
        "    cm = confusion_matrix(true_labels, zs_preds)\n",
        "    plt.figure(figsize=(6, 4))\n",
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n",
        "                xticklabels=['Negative', 'Neutral', 'Positive'],\n",
        "                yticklabels=['Negative', 'Neutral', 'Positive'])\n",
        "    plt.xlabel('Predicted')\n",
        "    plt.ylabel('True')\n",
        f"    plt.title('Zero-Shot Confusion Matrix ({display})')\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "else:\n",
        "    print(\"No ground-truth sentiment labels available for zero-shot evaluation.\")\n",
        "    zs_accuracy = zs_precision = zs_recall = zs_f1 = None"
    ])

    # ── N. Save Reports ───────────────────────────────────────────────
    md([
        "## N. Save All Reports to `output_txt/`\n",
        "Compile NER test results, SA test results, zero-shot metrics, and a\n",
        f"full summary into a text report and save it under `output_txt/`."
    ])
    code([
        "import os\n",
        "from datetime import datetime\n",
        "\n",
        "os.makedirs(\"output_txt\", exist_ok=True)\n",
        "\n",
        f"report_file = f\"output_txt/{display}_GEMINI_FLASH_EXTENDED_Report.txt\"\n",
        "\n",
        "with open(report_file, \"w\", encoding=\"utf-8\") as f:\n",
        "    f.write(\"=\" * 70 + \"\\n\")\n",
        f"    f.write(f\"  {display} – GEMINI FLASH EXTENDED – Full Report\\n\")\n",
        "    f.write(f\"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\")\n",
        "    f.write(\"=\" * 70 + \"\\n\\n\")\n",
        "\n",
        "    f.write(f\"Model: {model_id}\\n\")\n",
        "    f.write(f\"Training samples: {len(df_train)}\\n\")\n",
        "    f.write(f\"Test samples:     {len(df_test)}\\n\\n\")\n",
        "\n",
        "    # --- NER Results ---\n",
        "    f.write(\"-\" * 50 + \"\\n\")\n",
        "    f.write(\"NER Test Results\\n\")\n",
        "    f.write(\"-\" * 50 + \"\\n\")\n",
        "    if test_dataset_ner is not None:\n",
        "        for k, v in test_results_ner.items():\n",
        "            f.write(f\"  {k}: {v}\\n\")\n",
        "    else:\n",
        "        f.write(\"  No test NER dataset available.\\n\")\n",
        "    f.write(\"\\n\")\n",
        "\n",
        "    # --- SA Results ---\n",
        "    f.write(\"-\" * 50 + \"\\n\")\n",
        "    f.write(\"Sentiment Analysis Test Results\\n\")\n",
        "    f.write(\"-\" * 50 + \"\\n\")\n",
        "    for k, v in test_results_sa.items():\n",
        "        f.write(f\"  {k}: {v}\\n\")\n",
        "    f.write(\"\\n\")\n",
        "\n",
        "    # --- Zero-Shot Results ---\n",
        "    f.write(\"-\" * 50 + \"\\n\")\n",
        "    f.write(\"Zero-Shot Classification Results\\n\")\n",
        "    f.write(\"-\" * 50 + \"\\n\")\n",
        "    f.write(f\"  Zero-shot model: {zs_model_name}\\n\")\n",
        "    if zs_accuracy is not None:\n",
        "        f.write(f\"  Accuracy:  {zs_accuracy:.4f}\\n\")\n",
        "        f.write(f\"  Precision: {zs_precision:.4f}\\n\")\n",
        "        f.write(f\"  Recall:    {zs_recall:.4f}\\n\")\n",
        "        f.write(f\"  F1 Score:  {zs_f1:.4f}\\n\")\n",
        "    else:\n",
        "        f.write(\"  No ground-truth labels for evaluation.\\n\")\n",
        "    f.write(\"\\n\")\n",
        "\n",
        "    # --- Zero-shot per-sample predictions ---\n",
        "    f.write(\"-\" * 50 + \"\\n\")\n",
        "    f.write(\"Zero-Shot Per-Sample Predictions\\n\")\n",
        "    f.write(\"-\" * 50 + \"\\n\")\n",
        "    for _, row in df_zs.iterrows():\n",
        "        f.write(f\"  [{row['id']:>3}] {row['predicted_sentiment']:>8} \"\n",
        "                f\"(conf={row['confidence']:.4f}) | {row['review']}\\n\")\n",
        "    f.write(\"\\n\")\n",
        "\n",
        "    f.write(\"=\" * 70 + \"\\n\")\n",
        "    f.write(\"End of Report\\n\")\n",
        "    f.write(\"=\" * 70 + \"\\n\")\n",
        "\n",
        "print(f\"Report saved to: {report_file}\")\n",
        "print(\"\\nAll tasks completed successfully!\")"
    ])

    # ── assemble notebook ─────────────────────────────────────────────
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


# ── Model configurations ───────────────────────────────────────────────
MODELS = [
    {
        "model_name": "BANGLABERT",
        "model_id":   "csebuetnlp/banglabert",
        "short":      "banglabert",
        "display":    "BanglaBERT",
        "filename":   "Codes/BanglaBERT_GEMINI_FLASH_EXTENDED.ipynb",
    },
    {
        "model_name": "MBERT",
        "model_id":   "bert-base-multilingual-cased",
        "short":      "mbert",
        "display":    "mBERT (Bengali)",
        "filename":   "Codes/MBERT_BENGALI_GEMINI_FLASH_EXTENDED.ipynb",
    },
    {
        "model_name": "XLM_ROBERTA",
        "model_id":   "xlm-roberta-base",
        "short":      "xlmr",
        "display":    "XLM-RoBERTa",
        "filename":   "Codes/XLM_ROBERTA_GEMINI_FLASH_EXTENDED.ipynb",
    },
]

# ── Generate all notebooks ─────────────────────────────────────────────
os.makedirs("Codes", exist_ok=True)

for cfg in MODELS:
    nb = build_notebook(
        model_name=cfg["model_name"],
        model_id=cfg["model_id"],
        short=cfg["short"],
        display=cfg["display"],
    )
    path = cfg["filename"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"[OK] {path}  ({len(nb['cells'])} cells)")

print(f"\nAll {len(MODELS)} notebooks generated in Codes/")
import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source})

# Title
md([
    "# BanglaBERT: NER (Multi-word BIO) + Sentiment Analysis\n",
    "\n",
    "This notebook fine-tunes **BanglaBERT** (`csebuetnlp/banglabert`) for:\n",
    "1. **Named Entity Recognition (NER)** - extracting product entities using BIO tagging\n",
    "2. **Sentiment Analysis (SA)** - classifying reviews as Negative / Neutral / Positive\n",
    "\n",
    "**Training data:** Gemini 3.5-Flash Synthetic Dataset  \n",
    "**Test data:** Test.csv"
])

# A. Install
md([
    "## A. Install Required Libraries\n",
    "Install Hugging Face `transformers`, `datasets`, `evaluate`, `seqeval` (for token-level NER metrics),\n",
    "`torch` as the deep-learning backend, `sentencepiece` for sub-word tokenization,\n",
    "and `accelerate>=1.1.0` (required by `Trainer` in newer transformers)."
])
code([
    "# -*- coding: utf-8 -*-\n",
    "!pip install datasets tokenizers seqeval -q\n",
    "!pip install transformers\n",
    "!pip install torch\n",
    "!pip install sentencepiece\n",
    "!pip install evaluate\n",
    "!pip install 'accelerate>=1.1.0' -q"
])

# B. Import
md([
    "## B. Import Libraries\n",
    "Import standard data-science and NLP libraries. We disable Weights & Biases logging\n",
    "via the `WANDB_DISABLED` environment variable so training runs stay local."
])
code([
    "import os\n",
    "os.environ[\"WANDB_DISABLED\"] = \"true\"\n",
    "\n",
    "import re\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import (\n",
    "    precision_score, recall_score, f1_score, accuracy_score,\n",
    "    confusion_matrix, roc_curve, auc\n",
    ")\n",
    "from datasets import Dataset\n",
    "from transformers import (\n",
    "    AutoTokenizer, AutoConfig, AutoModelForTokenClassification,\n",
    "    AutoModelForSequenceClassification, TrainingArguments, Trainer,\n",
    "    DataCollatorForTokenClassification, DataCollatorWithPadding\n",
    ")\n",
    "import evaluate"
])

# C. Load Data
md([
    "## C. Load Datasets\n",
    "Load the **Gemini 3.5-Flash Synthetic Dataset** as the training set and the **Test** dataset\n",
    "from the `Synthetic Datasets/` folder. Both CSVs contain columns: `id`, `review`, `sentiment`,\n",
    "and `bio_tagged_review`."
])
code([
    "TRAIN_CSV = 'Synthetic Datasets/Gemini 3.5-Flash_Synthetic_Dataset.csv'\n",
    "TEST_CSV  = 'Synthetic Datasets/Test.csv'\n",
    "\n",
    "df_train = pd.read_csv(TRAIN_CSV)\n",
    "df_test  = pd.read_csv(TEST_CSV)\n",
    "\n",
    "print(f\"Training samples: {len(df_train)}\")\n",
    "print(f\"Test samples:     {len(df_test)}\")\n",
    "df_train.head()"
])

# D. Parse BIO
md([
    "## D. Parse BIO Tags into Tokens & NER Labels\n",
    "Each `bio_tagged_review` string contains tokens annotated as `token-B` or `token-I`.\n",
    "We parse them into parallel lists of **tokens** and **BIO labels** (`B-Product`, `I-Product`, `O`).\n",
    "For the test set, if `bio_tagged_review` is missing we fall back to whitespace tokenization."
])
code([
    "def parse_bio_tags(bio_text):\n",
    "    tokens, labels = [], []\n",
    "    for part in bio_text.split():\n",
    "        match = re.match(r'^(.*?)(?:[,\\s\u0964]*)-(B|I)$', part)\n",
    "        if match:\n",
    "            token, tag = match.group(1), match.group(2)\n",
    "            labels.append('B-Product' if tag == 'B' else 'I-Product')\n",
    "        else:\n",
    "            token = part\n",
    "            labels.append('O')\n",
    "        tokens.append(token)\n",
    "    return tokens, labels\n",
    "\n",
    "# Parse training set\n",
    "df_train['tokens']   = df_train['bio_tagged_review'].apply(lambda x: parse_bio_tags(x)[0])\n",
    "df_train['ner_tags'] = df_train['bio_tagged_review'].apply(lambda x: parse_bio_tags(x)[1])\n",
    "\n",
    "# Parse test set (fall back to plain split if no BIO column)\n",
    "if 'bio_tagged_review' in df_test.columns:\n",
    "    df_test['tokens']   = df_test['bio_tagged_review'].apply(lambda x: parse_bio_tags(x)[0])\n",
    "    df_test['ner_tags'] = df_test['bio_tagged_review'].apply(lambda x: parse_bio_tags(x)[1])\n",
    "else:\n",
    "    df_test['tokens'] = df_test['review'].apply(lambda x: x.split())\n",
    "\n",
    "print(df_train[['review', 'tokens', 'ner_tags']].iloc[0])"
])

# E. Label Mappings
md([
    "## E. Define Label Mappings\n",
    "Create bidirectional mappings between label strings and integer IDs.\n",
    "Our NER scheme has three labels: `O` (non-entity), `B-Product` (beginning of product),\n",
    "and `I-Product` (inside product)."
])
code([
    "label_list = ['O', 'B-Product', 'I-Product']\n",
    "id2label   = {i: label for i, label in enumerate(label_list)}\n",
    "label2id   = {label: i for i, label in enumerate(label_list)}\n",
    "num_labels = len(label_list)\n",
    "\n",
    "print(f\"Labels ({num_labels}): {label_list}\")\n",
    "print(f\"label2id: {label2id}\")"
])

# F. Split
md([
    "## F. Split Training Data (80/20)\n",
    "Create a stratified 80/20 train/validation split so we can monitor overfitting during training.\n",
    "Stratification on `sentiment` ensures the class distribution stays consistent across splits."
])
code([
    "X = df_train[['id', 'review', 'tokens', 'ner_tags']]\n",
    "y = df_train['sentiment']\n",
    "\n",
    "X_train, X_val, y_train, y_val = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42, stratify=y\n",
    ")\n",
    "X_train = X_train.reset_index(drop=True)\n",
    "X_val   = X_val.reset_index(drop=True)\n",
    "\n",
    "print(f\"Train size: {len(X_train)},  Validation size: {len(X_val)}\")"
])

# G. NER Tokenizer
md([
    "## G. NER Tokenizer & Label Alignment (BanglaBERT)\n",
    "Load the **BanglaBERT** tokenizer (`csebuetnlp/banglabert`). Because sub-word tokenization\n",
    "splits a single word into multiple tokens, we align NER labels using `word_ids()`:\n",
    "only the **first sub-token** of each word receives the true label; the rest get `-100`\n",
    "(ignored during loss computation)."
])
code([
    "BANGLABERT = \"csebuetnlp/banglabert\"\n",
    "\n",
    "tokenizer_ner = AutoTokenizer.from_pretrained(BANGLABERT)\n",
    "\n",
    "def tokenize_and_align_labels(examples, tokenizer, label2id):\n",
    "    tokenized_inputs = tokenizer(\n",
    "        examples['tokens'],\n",
    "        truncation=True,\n",
    "        is_split_into_words=True,\n",
    "        padding=False\n",
    "    )\n",
    "    all_labels = []\n",
    "    for i, label_seq in enumerate(examples['ner_tags']):\n",
    "        word_ids = tokenized_inputs.word_ids(batch_index=i)\n",
    "        previous_word_idx = None\n",
    "        label_ids = []\n",
    "        for word_idx in word_ids:\n",
    "            if word_idx is None:\n",
    "                label_ids.append(-100)\n",
    "            elif word_idx != previous_word_idx:\n",
    "                label_ids.append(label2id[label_seq[word_idx]])\n",
    "            else:\n",
    "                label_ids.append(-100)\n",
    "            previous_word_idx = word_idx\n",
    "        all_labels.append(label_ids)\n",
    "    tokenized_inputs['labels'] = all_labels\n",
    "    return tokenized_inputs\n",
    "\n",
    "def create_ner_dataset(df, tokenizer, label2id):\n",
    "    tokenized = tokenize_and_align_labels(\n",
    "        {'tokens': df['tokens'].tolist(), 'ner_tags': df['ner_tags'].tolist()},\n",
    "        tokenizer, label2id\n",
    "    )\n",
    "    return Dataset.from_dict({\n",
    "        'input_ids':      tokenized['input_ids'],\n",
    "        'attention_mask': tokenized['attention_mask'],\n",
    "        'labels':         tokenized['labels'],\n",
    "        'tokens':         df['tokens'].tolist(),\n",
    "        'ner_tags':       df['ner_tags'].tolist(),\n",
    "        'id':             df['id'].tolist() if 'id' in df else list(range(len(df)))\n",
    "    })\n",
    "\n",
    "train_dataset_ner = create_ner_dataset(X_train, tokenizer_ner, label2id)\n",
    "val_dataset_ner   = create_ner_dataset(X_val,   tokenizer_ner, label2id)\n",
    "\n",
    "if 'tokens' in df_test.columns and 'ner_tags' in df_test.columns:\n",
    "    test_dataset_ner = create_ner_dataset(df_test, tokenizer_ner, label2id)\n",
    "else:\n",
    "    test_dataset_ner = None\n",
    "\n",
    "print(f\"NER datasets created -- train: {len(train_dataset_ner)}, val: {len(val_dataset_ner)}\")"
])

# H. Train NER
md([
    "## H. Train the NER Model (BanglaBERT Token Classification)\n",
    "Combine the train + validation splits into one full training set, then fine-tune\n",
    "**BanglaBERT** for token classification (3 epochs). We evaluate on the held-out\n",
    "validation set using the `seqeval` metric and save the best checkpoint.\n",
    "\n",
    "> **Note:** BanglaBERT uses an ELECTRA architecture internally, so the load report\n",
    "> will show UNEXPECTED (discriminator head) and MISSING (classifier) keys.\n",
    "> This is normal -- the new classifier layer is randomly initialized for fine-tuning."
])
code([
    "print(\"=== Training NER model on full training data ===\\n\")\n",
    "\n",
    "# Merge train + val for final training\n",
    "full_train = pd.concat([X_train, X_val], ignore_index=True)\n",
    "full_train['sentiment'] = pd.concat([y_train, y_val], ignore_index=True)\n",
    "full_train_ds = create_ner_dataset(full_train, tokenizer_ner, label2id)\n",
    "\n",
    "# Build model from BanglaBERT pretrained weights\n",
    "config = AutoConfig.from_pretrained(BANGLABERT)\n",
    "config.num_labels = num_labels\n",
    "config.id2label   = id2label\n",
    "config.label2id   = label2id\n",
    "model_ner = AutoModelForTokenClassification.from_pretrained(\n",
    "    BANGLABERT, config=config\n",
    ")\n",
    "\n",
    "args_ner = TrainingArguments(\n",
    "    output_dir=\"ner_final\",\n",
    "    eval_strategy=\"epoch\",\n",
    "    learning_rate=3e-5,\n",
    "    per_device_train_batch_size=16,\n",
    "    per_device_eval_batch_size=16,\n",
    "    num_train_epochs=3,\n",
    "    weight_decay=0.01,\n",
    "    logging_steps=50,\n",
    "    save_strategy=\"epoch\",\n",
    "    load_best_model_at_end=True,\n",
    "    metric_for_best_model=\"eval_f1\",\n",
    ")\n",
    "\n",
    "data_collator_ner = DataCollatorForTokenClassification(tokenizer_ner)\n",
    "metric_ner = evaluate.load(\"seqeval\")\n",
    "\n",
    "def compute_metrics_ner(eval_preds):\n",
    "    pred_logits, labels = eval_preds\n",
    "    pred_logits = np.argmax(pred_logits, axis=2)\n",
    "    predictions = [\n",
    "        [label_list[p] for (p, l) in zip(pred, lab) if l != -100]\n",
    "        for pred, lab in zip(pred_logits, labels)\n",
    "    ]\n",
    "    references = [\n",
    "        [label_list[l] for (p, l) in zip(pred, lab) if l != -100]\n",
    "        for pred, lab in zip(pred_logits, labels)\n",
    "    ]\n",
    "    results = metric_ner.compute(predictions=predictions, references=references)\n",
    "    return {\n",
    "        \"precision\": results[\"overall_precision\"],\n",
    "        \"recall\":    results[\"overall_recall\"],\n",
    "        \"f1\":        results[\"overall_f1\"],\n",
    "        \"accuracy\":  results[\"overall_accuracy\"],\n",
    "    }\n",
    "\n",
    "trainer_ner = Trainer(\n",
    "    model=model_ner,\n",
    "    args=args_ner,\n",
    "    train_dataset=full_train_ds,\n",
    "    eval_dataset=val_dataset_ner,\n",
    "    data_collator=data_collator_ner,\n",
    "    tokenizer=tokenizer_ner,\n",
    "    compute_metrics=compute_metrics_ner\n",
    ")\n",
    "\n",
    "trainer_ner.train()\n",
    "\n",
    "# Evaluate on test set if available\n",
    "if test_dataset_ner is not None:\n",
    "    test_results_ner = trainer_ner.evaluate(test_dataset_ner)\n",
    "    print(\"\\n=== NER Test Results ===\")\n",
    "    print(test_results_ner)\n",
    "\n",
    "model_ner.save_pretrained(\"ner_model_final\")\n",
    "tokenizer_ner.save_pretrained(\"ner_tokenizer\")\n",
    "print(\"\\nNER model saved.\")"
])

# I. SA Prep
md([
    "## I. Sentiment Analysis - Tokenize & Prepare Datasets (BanglaBERT)\n",
    "Load a separate **BanglaBERT** tokenizer for sequence classification.\n",
    "Tokenize the raw review text and create `Dataset` objects for train, validation, and test splits."
])
code([
    "sa_tokenizer = AutoTokenizer.from_pretrained(BANGLABERT)\n",
    "\n",
    "def tokenize_sa(reviews, tokenizer):\n",
    "    return tokenizer(reviews, truncation=True, padding=False)\n",
    "\n",
    "# Prepare text & labels for each split\n",
    "X_train_sa      = X_train['review'].tolist()\n",
    "y_train_sa      = y_train.tolist()\n",
    "X_val_sa        = X_val['review'].tolist()\n",
    "y_val_sa        = y_val.tolist()\n",
    "full_train_sa   = full_train['review'].tolist()\n",
    "full_train_sa_y = full_train['sentiment'].tolist()\n",
    "\n",
    "# Tokenize\n",
    "train_enc_sa      = tokenize_sa(X_train_sa,    sa_tokenizer)\n",
    "val_enc_sa        = tokenize_sa(X_val_sa,      sa_tokenizer)\n",
    "full_train_enc_sa = tokenize_sa(full_train_sa,  sa_tokenizer)\n",
    "\n",
    "# Build HuggingFace Datasets\n",
    "train_sa_dataset = Dataset.from_dict({\n",
    "    'input_ids':      train_enc_sa['input_ids'],\n",
    "    'attention_mask': train_enc_sa['attention_mask'],\n",
    "    'labels':         y_train_sa,\n",
    "})\n",
    "val_sa_dataset = Dataset.from_dict({\n",
    "    'input_ids':      val_enc_sa['input_ids'],\n",
    "    'attention_mask': val_enc_sa['attention_mask'],\n",
    "    'labels':         y_val_sa,\n",
    "})\n",
    "full_train_sa_dataset = Dataset.from_dict({\n",
    "    'input_ids':      full_train_enc_sa['input_ids'],\n",
    "    'attention_mask': full_train_enc_sa['attention_mask'],\n",
    "    'labels':         full_train_sa_y,\n",
    "})\n",
    "\n",
    "print(f\"SA datasets created -- train: {len(train_sa_dataset)}, val: {len(val_sa_dataset)}\")"
])

# J. Train SA
md([
    "## J. Train the Sentiment Analysis Model (BanglaBERT Sequence Classification)\n",
    "Fine-tune **BanglaBERT** for 3-class sentiment classification (Negative / Neutral / Positive)\n",
    "on the full training set. The best checkpoint (by validation accuracy) is kept."
])
code([
    "print(\"=== Training final Sentiment Analysis model ===\\n\")\n",
    "\n",
    "model_sa = AutoModelForSequenceClassification.from_pretrained(\n",
    "    BANGLABERT,\n",
    "    num_labels=3,\n",
    "    id2label={0: \"NEGATIVE\", 1: \"NEUTRAL\", 2: \"POSITIVE\"},\n",
    "    label2id={\"NEGATIVE\": 0, \"NEUTRAL\": 1, \"POSITIVE\": 2}\n",
    ")\n",
    "\n",
    "args_sa = TrainingArguments(\n",
    "    output_dir=\"sa_final\",\n",
    "    eval_strategy=\"epoch\",\n",
    "    learning_rate=3e-5,\n",
    "    per_device_train_batch_size=16,\n",
    "    per_device_eval_batch_size=16,\n",
    "    num_train_epochs=3,\n",
    "    weight_decay=0.01,\n",
    "    logging_steps=50,\n",
    "    save_strategy=\"epoch\",\n",
    "    load_best_model_at_end=True,\n",
    "    metric_for_best_model=\"eval_accuracy\",\n",
    ")\n",
    "\n",
    "data_collator_sa = DataCollatorWithPadding(sa_tokenizer)\n",
    "\n",
    "def compute_metrics_sa(eval_pred):\n",
    "    preds  = eval_pred.predictions.argmax(-1)\n",
    "    labels = eval_pred.label_ids\n",
    "    return {\n",
    "        'precision': precision_score(labels, preds, average='weighted'),\n",
    "        'recall':    recall_score(labels, preds, average='weighted'),\n",
    "        'f1':        f1_score(labels, preds, average='weighted'),\n",
    "        'accuracy':  accuracy_score(labels, preds)\n",
    "    }\n",
    "\n",
    "trainer_sa = Trainer(\n",
    "    model=model_sa,\n",
    "    args=args_sa,\n",
    "    train_dataset=full_train_sa_dataset,\n",
    "    eval_dataset=val_sa_dataset,\n",
    "    tokenizer=sa_tokenizer,\n",
    "    data_collator=data_collator_sa,\n",
    "    compute_metrics=compute_metrics_sa\n",
    ")\n",
    "\n",
    "trainer_sa.train()\n",
    "print(\"\\nSA model training complete.\")"
])

# K. Eval Test
md([
    "## K. Evaluate Sentiment Model on Test Set\n",
    "Tokenize the test-set reviews and run evaluation. If the test CSV lacks a `sentiment`\n",
    "column, dummy labels are used (accuracy will be meaningless but the pipeline won't crash)."
])
code([
    "test_reviews  = df_test['review'].tolist()\n",
    "test_enc_sa   = tokenize_sa(test_reviews, sa_tokenizer)\n",
    "\n",
    "if 'sentiment' in df_test.columns:\n",
    "    test_labels = df_test['sentiment'].tolist()\n",
    "else:\n",
    "    test_labels = [0] * len(df_test)\n",
    "    print(\"Warning: No sentiment column in test set - using dummy labels.\")\n",
    "\n",
    "test_sa_dataset = Dataset.from_dict({\n",
    "    'input_ids':      test_enc_sa['input_ids'],\n",
    "    'attention_mask': test_enc_sa['attention_mask'],\n",
    "    'labels':         test_labels\n",
    "})\n",
    "\n",
    "test_results_sa = trainer_sa.evaluate(test_sa_dataset)\n",
    "print(\"\\n=== SA Test Results ===\")\n",
    "print(test_results_sa)\n",
    "\n",
    "model_sa.save_pretrained(\"sa_model_final\")\n",
    "sa_tokenizer.save_pretrained(\"sa_tokenizer\")\n",
    "print(\"SA model saved.\")"
])

# L. Inference
md([
    "## L. Inference Example\n",
    "Demonstrate end-to-end inference on a sample Bangla review:\n",
    "1. **NER pipeline** - extract product entities from the text.\n",
    "2. **Sentiment classification** - predict the overall sentiment (Negative / Neutral / Positive)."
])
code([
    "from transformers import pipeline\n",
    "\n",
    "# --- NER inference ---\n",
    "ner_pipe = pipeline(\n",
    "    \"ner\",\n",
    "    model=\"ner_model_final\",\n",
    "    tokenizer=\"ner_tokenizer\",\n",
    "    aggregation_strategy=\"simple\"\n",
    ")\n",
    "\n",
    "sample_review = \"\u09ab\u09c7\u09b8 \u0993\u09af\u09bc\u09be\u09b6 \u099f\u09be \u0985\u09a8\u09c7\u0995 \u09b8\u09c1\u09a8\u09cd\u09a6\u09b0, \u0985\u09b8\u0982\u0996\u09cd\u09af \u09a7\u09a8\u09cd\u09af\u09ac\u09be\u09a6 \u09a1\u09c7\u09b2\u09bf\u09ad\u09be\u09b0\u09bf \u0996\u09c1\u09ac \u09a6\u09cd\u09b0\u09c1\u09a4 \u09a6\u09c7\u09af\u09bc\u09be\u09b0 \u099c\u09a8\u09cd\u09af\u0964\"\n",
    "print(\"Sample review:\", sample_review)\n",
    "print(\"NER results:\", ner_pipe(sample_review))\n",
    "\n",
    "# --- Sentiment inference ---\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "model_sa.to(device)\n",
    "\n",
    "enc = sa_tokenizer([sample_review], padding=True, truncation=True, return_tensors=\"pt\")\n",
    "enc = {k: v.to(device) for k, v in enc.items()}\n",
    "\n",
    "with torch.no_grad():\n",
    "    logits = model_sa(**enc).logits\n",
    "    pred = logits.argmax(dim=1).item()\n",
    "\n",
    "sentiment_map = {0: \"Negative\", 1: \"Neutral\", 2: \"Positive\"}\n",
    "print(f\"\\nPredicted sentiment: {sentiment_map[pred]}\")\n",
    "\n",
    "print(\"\\nAll tasks completed successfully!\")"
])

# Write notebook
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("BanglaBert.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("BanglaBert.ipynb created successfully!")
