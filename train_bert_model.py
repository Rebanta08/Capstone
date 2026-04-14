#!/usr/bin/env python3
# ============================================================
# BERT Fine-Tuning for IT Service Ticket Classification
# Trains a deep transformer model and saves artifacts
# ============================================================

# Install required packages
import subprocess
import sys

packages = [
    "torch",
    "transformers",
    "pandas",
    "scikit-learn",
    "tqdm"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", package])

import os
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# ============================================================
# 1) LOAD AND PREPARE DATA
# ============================================================
print("Loading dataset...")
df = pd.read_csv("all_tickets_processed_improved_v3.csv")

# Clean data
df["Document"] = df["Document"].fillna("").astype(str)
df["Topic_group"] = df["Topic_group"].fillna("").astype(str)

df = df[
    (df["Document"].str.strip() != "") &
    (df["Topic_group"].str.strip() != "")
].copy()

# Rename for consistency
df = df.rename(columns={"Topic_group": "Actual"})

# Get unique classes and create label mapping
classes = sorted(df["Actual"].unique().tolist())
label2id = {cls: idx for idx, cls in enumerate(classes)}
id2label = {idx: cls for cls, idx in label2id.items()}
num_labels = len(classes)

print(f"\nDataset info:")
print(f"Total samples: {len(df)}")
print(f"Number of classes: {num_labels}")
print(f"Classes: {classes}")

# Split into train/val
X = df["Document"].values
y = df["Actual"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)}")
print(f"Val size: {len(X_val)}")

# ============================================================
# 2) CREATE PYTORCH DATASET
# ============================================================
class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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
# 3) INITIALIZE MODEL AND TRAINING
# ============================================================
print("\nInitializing BERT model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels
)
model.to(device)

# Convert labels to indices
y_train_idx = [label2id[label] for label in y_train]
y_val_idx = [label2id[label] for label in y_val]

# Create datasets and loaders
train_dataset = TicketDataset(X_train, y_train, tokenizer)
val_dataset = TicketDataset(X_val, y_val, tokenizer)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# ============================================================
# 4) TRAINING LOOP
# ============================================================
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
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
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy, predictions, true_labels

# Train for 4 epochs (more data = potentially better convergence)
print("\nTraining BERT model (4 epochs)...")
for epoch in range(4):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1}/3")
    print(f"{'='*50}")
    
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f"\nTrain loss: {train_loss:.4f}")
    
    val_acc, _, _ = eval_epoch(model, val_loader, device)
    print(f"Val accuracy: {val_acc:.4f}")

# ============================================================
# 5) FINAL EVALUATION
# ============================================================
print("\n" + "="*50)
print("FINAL EVALUATION ON VALIDATION SET")
print("="*50)

val_acc, val_predictions, val_true = eval_epoch(model, val_loader, device)

print(f"\nValidation Accuracy: {val_acc:.4f}")
print(f"\nClassification Report:\n")
print(classification_report(val_true, val_predictions, target_names=classes))

# Confusion matrix
print("\nTop 8 classes confusion matrix:")
cm = confusion_matrix(val_true, val_predictions, labels=range(num_labels))
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
print(cm_df)

# ============================================================
# 6) SAVE MODEL AND ARTIFACTS
# ============================================================
print("\n" + "="*50)
print("SAVING MODEL AND ARTIFACTS")
print("="*50)

# Save model and tokenizer
model_path = "bert_ticket_classifier"
os.makedirs(model_path, exist_ok=True)

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print(f"✓ Model saved to: {model_path}/")

# Save metadata
metadata = {
    "model_type": "bert-base-uncased",
    "num_labels": num_labels,
    "classes": classes,
    "label2id": label2id,
    "id2label": id2label,
    "validation_accuracy": float(val_acc),
    "num_classes": num_labels,
    "max_length": 128,
    "batch_size": batch_size,
    "epochs_trained": 4,
    "learning_rate": 2e-5
}

with open(f"{model_path}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Metadata saved to: {model_path}/metadata.json")

# ============================================================
# 7) TEST PREDICTIONS WITH CONFIDENCE
# ============================================================
print("\n" + "="*50)
print("SAMPLE PREDICTIONS WITH CONFIDENCE")
print("="*50)

def predict_ticket(text, model, tokenizer, device, id2label):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]
        
        pred_idx = torch.argmax(probabilities).item()
        confidence = probabilities[pred_idx].item()
        
        return id2label[pred_idx], confidence, probabilities.cpu().numpy()

test_tickets = [
    "Users unable to access VPN after password reset, login failure",
    "Printer offline, cannot print payroll documents",
    "Outlook mailbox full, messages bouncing back",
    "Laptop needs new monitor and keyboard",
    "New employee needs shared folder and storage access"
]

print("\nTest predictions:\n")
for ticket in test_tickets:
    pred_class, confidence, probs = predict_ticket(
        ticket, model, tokenizer, device, id2label
    )
    print(f"Text: {ticket[:60]}...")
    print(f"Predicted: {pred_class} (confidence: {confidence:.4f})\n")

print("✓ Model ready for deployment!")
