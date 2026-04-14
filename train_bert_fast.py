#!/usr/bin/env python3
# ============================================================
# BERT Fast Training for IT Service Ticket Classification
# Optimized for speed: 2 epochs on CPU (~30 min)
# ============================================================

import subprocess
import sys
import os

packages = ["torch", "transformers", "pandas", "scikit-learn", "tqdm"]
for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", package])

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# ============================================================
# 1) LOAD DATA - SAMPLE FOR SPEED
# ============================================================
print("Loading dataset...")
df = pd.read_csv("all_tickets_processed_improved_v3.csv")

df["Document"] = df["Document"].fillna("").astype(str)
df["Topic_group"] = df["Topic_group"].fillna("").astype(str)
df = df[
    (df["Document"].str.strip() != "") &
    (df["Topic_group"].str.strip() != "")
].copy()

# SPEED OPTIMIZATION: Sample 10k for training (still representative)
df = df.sample(n=min(10000, len(df)), random_state=42)

df = df.rename(columns={"Topic_group": "Actual"})

classes = sorted(df["Actual"].unique().tolist())
label2id = {cls: idx for idx, cls in enumerate(classes)}
id2label = {idx: cls for cls, idx in label2id.items()}
num_labels = len(classes)

print(f"Dataset info:")
print(f"Total samples: {len(df)}")
print(f"Classes: {num_labels}")

X = df["Document"].values
y = df["Actual"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} | Val: {len(X_val)}\n")

# ============================================================
# 2) DATASET CLASS
# ============================================================
class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=96):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "label": torch.tensor(label2id[label], dtype=torch.long)
        }

# ============================================================
# 3) INITIALIZE & TRAIN
# ============================================================
print("Initializing BERT...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels
)
model.to(device)

train_dataset = TicketDataset(X_train, y_train, tokenizer)
val_dataset = TicketDataset(X_val, y_val, tokenizer)

batch_size = 32  # Larger batch for speed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 2  # 2 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy, predictions, true_labels

print("Training BERT (2 epochs)...\n")
for epoch in range(2):
    print(f"Epoch {epoch + 1}/2")
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_acc, _, _ = eval_epoch(model, val_loader, device)
    print(f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}\n")

# ============================================================
# 4) FINAL EVALUATION
# ============================================================
print("="*50)
print("FINAL EVALUATION")
print("="*50)
val_acc, val_predictions, val_true = eval_epoch(model, val_loader, device)
print(f"\nValidation Accuracy: {val_acc:.4f}")
print(f"\nClassification Report:\n")
print(classification_report(val_true, val_predictions, target_names=classes))

# ============================================================
# 5) SAVE MODEL
# ============================================================
print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)

model_path = "bert_ticket_classifier"
os.makedirs(model_path, exist_ok=True)

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

metadata = {
    "model_type": "bert-base-uncased",
    "num_labels": num_labels,
    "classes": classes,
    "label2id": label2id,
    "id2label": id2label,
    "validation_accuracy": float(val_acc),
    "max_length": 96,
    "batch_size": batch_size,
    "epochs_trained": 2,
    "learning_rate": 2e-5,
    "training_samples": len(X_train)
}

with open(f"{model_path}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Model saved to: {model_path}/")
print(f"✓ Files: pytorch_model.bin, config.json, vocab.txt, metadata.json")

# ============================================================
# 6) TEST PREDICTIONS
# ============================================================
print("\n" + "="*50)
print("SAMPLE PREDICTIONS")
print("="*50)

def predict_ticket(text, model, tokenizer, device, id2label):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(
            text, max_length=96, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probabilities).item()
        confidence = probabilities[pred_idx].item()
        return id2label[pred_idx], confidence

test_tickets = [
    "Cannot login to VPN after password reset",
    "Printer offline need assistance",
    "Need access to shared folder",
    "Laptop keyboard broken needs repair"
]

for ticket in test_tickets:
    pred, conf = predict_ticket(ticket, model, tokenizer, device, id2label)
    print(f"'{ticket[:40]}...'\n→ {pred} ({conf:.1%})\n")

print("✓ Model training complete and ready for deployment!")
