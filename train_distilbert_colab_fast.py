# Fast ALBERT-base Training for IT Ticket Classification (Colab-ready, small data)
# - Uses only 25% of data
# - 3 epochs
# - 5% data augmentation
# - All other optimizations preserved

!pip install torch transformers pandas scikit-learn tqdm --quiet

import os
import json
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AlbertTokenizer, AlbertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Path to your CSV (upload to Colab or Drive)
CSV_PATH = 'all_tickets_processed_improved_v3.csv'  # Update if needed

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=0.25, random_state=42).reset_index(drop=True)  # Use only 25% of data
print(f"Using {len(df)} samples (25% of full dataset)")
df["Document"] = df["Document"].fillna("").astype(str)
df["Topic_group"] = df["Topic_group"].fillna("").astype(str)
df = df[(df["Document"].str.strip() != "") & (df["Topic_group"].str.strip() != "")].copy()
df = df.rename(columns={"Topic_group": "Actual"})

classes = sorted(df["Actual"].unique().tolist())
label2id = {cls: idx for idx, cls in enumerate(classes)}
id2label = {idx: cls for cls, idx in label2id.items()}
num_labels = len(classes)

print(f"\nDataset info:")
print(f"Total samples: {len(df)}")
print(f"Number of classes: {num_labels}")
print(f"Classes: {classes}")

X = df["Document"].values
y = df["Actual"].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Data augmentation (5% dropout)
print("Applying data augmentation (5% dropout)...")
def augment_text(text, prob=0.05):
    words = text.split()
    if len(words) > 5:
        augmented = [w for w in words if random.random() > prob]
        return " ".join(augmented) if augmented else text
    return text
X_train_aug = [augment_text(text) for text in X_train]
X_train = np.concatenate([X_train, X_train_aug])
y_train = np.concatenate([y_train, y_train])

print(f"After augmentation - Train size: {len(X_train)}")

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=num_labels)
model.to(device)

y_train_idx = np.array([label2id[label] for label in y_train])
y_val_idx = np.array([label2id[label] for label in y_val])

train_dataset = TicketDataset(X_train, y_train, tokenizer)
val_dataset = TicketDataset(X_val, y_val, tokenizer)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_idx), y=y_train_idx)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)
print(f"Class weights: {class_weights}")
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = AdamW(model.parameters(), lr=3e-5)
num_epochs = 3
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

def train_epoch(model, loader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total

def validate_epoch(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels

print("\nTraining...")
best_val_accuracy = 0
patience = 2
patience_counter = 0
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, loss_fn)
    val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, device, loss_fn)
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
        print(f"  ✓ Best model saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

print("\nLoading Best Model...")
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, device, loss_fn)
print(f"\nBest Model - Val Accuracy: {val_acc:.4f}")
print("\nClassification Report:")
print(classification_report(val_labels, val_preds, target_names=classes, digits=3))

os.makedirs("albert_ticket_classifier", exist_ok=True)
model.save_pretrained("albert_ticket_classifier")
tokenizer.save_pretrained("albert_ticket_classifier")
metadata = {
    "classes": classes,
    "label2id": label2id,
    "id2label": id2label,
    "num_labels": num_labels,
    "val_accuracy": float(val_acc),
    "max_length": 128,
    "model_type": "albert-base-v2",
    "training_config": {
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": 3e-5,
        "warmup_ratio": 0.1,
        "augmentation": "5% word dropout",
        "class_weights": "balanced"
    }
}
with open("albert_ticket_classifier/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("✓ Model saved to albert_ticket_classifier/")
print(f"✓ Validation Accuracy: {val_acc:.1%}")
print("Done!")
