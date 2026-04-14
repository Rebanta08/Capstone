#!/usr/bin/env python3
# ============================================================
# BERT Minimal Training - Ultra-Fast
# Creates and saves a fine-tuned BERT model quickly
# ============================================================

import subprocess
import sys

packages = ["torch", "transformers", "pandas", "scikit-learn"]
for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", package])

import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("Loading and preparing data...")
df = pd.read_csv("all_tickets_processed_improved_v3.csv")
df["Document"] = df["Document"].fillna("").astype(str)
df["Topic_group"] = df["Topic_group"].fillna("").astype(str)
df = df[(df["Document"].str.strip() != "") & (df["Topic_group"].str.strip() != "")].copy()

# Only sample 2000 for ultra-fast training
df = df.sample(n=2000, random_state=42)
df = df.rename(columns={"Topic_group": "Actual"})

classes = sorted(df["Actual"].unique().tolist())
label2id = {cls: idx for idx, cls in enumerate(classes)}
id2label = {idx: cls for cls, idx in label2id.items()}

print(f"Classes: {len(classes)} | Samples: {len(df)}")

X = df["Document"].values
y = df["Actual"].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple tokenization without dataset class for speed
print("Initializing model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(classes))
model.to(device)

# Tokenize
def tokenize_batch(texts, labels):
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=64, return_tensors="pt")
    return encodings, torch.tensor([label2id[l] for l in labels], dtype=torch.long)

# Quick training - just 1 epoch on small subset
print("\nTraining on 2000 samples (1 epoch)...")
optimizer = AdamW(model.parameters(), lr=2e-5)

model.train()
train_texts = X_train[:500]  # Just first 500 for speed
train_labels = y_train[:500]

print(f"Training on {len(train_texts)} samples...")
for i in range(0, len(train_texts), 64):
    batch_texts = train_texts[i:i+64]
    batch_labels = train_labels[i:i+64]
    
    encodings, labels = tokenize_batch(batch_texts, batch_labels)
    
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    
    loss.backward()
    optimizer.step()
    
    if (i // 64) % 2 == 0:
        print(f"  Batch {i//64}, Loss: {loss.item():.4f}")

# Evaluation on validation
print("\nEvaluating...")
model.eval()
val_predictions = []
val_true = []

with torch.no_grad():
    for i in range(0, len(X_val), 64):
        batch_texts = X_val[i:i+64]
        batch_labels = y_val[i:i+64]
        
        encodings, labels = tokenize_batch(batch_texts, batch_labels)
        
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        val_predictions.extend(preds)
        val_true.extend(labels.numpy())

val_acc = accuracy_score(val_true, val_predictions)
print(f"\nValidation Accuracy: {val_acc:.4f}")
print(f"\nClassification Report:\n{classification_report(val_true, val_predictions, target_names=classes)}")

# Save model
print("\nSaving model...")
model_path = "bert_ticket_classifier"
os.makedirs(model_path, exist_ok=True)
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

metadata = {
    "model_type": "bert-base-uncased",
    "num_labels": len(classes),
    "classes": classes,
    "label2id": label2id,
    "id2label": id2label,
    "validation_accuracy": float(val_acc),
    "max_length": 64,
    "epochs_trained": 1,
    "training_samples": 500
}

with open(f"{model_path}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Model saved to {model_path}/")
print("✓ Ready for deployment!")
