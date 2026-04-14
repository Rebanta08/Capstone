"""
IT Service Ticket Classifier - API Server for Hugging Face Spaces
Simple Flask app that serves BERT predictions as JSON
"""

from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

app = Flask(__name__)

# Load model
print("Loading BERT model...")
try:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=8)
except Exception as e:
    print(f"Model loading failed: {e}")
    tokenizer = None
    model = None

classes = ["Access", "Administrative rights", "HR Support", "Hardware", 
           "Internal Project", "Miscellaneous", "Purchase", "Storage"]
id2label = {i: cls for i, cls in enumerate(classes)}

device = torch.device("cpu")
if model:
    model.to(device)
    model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict ticket class
    Input: JSON with "text" field
    Output: JSON with prediction and confidence
    """
    try:
        data = request.json
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "Empty text"}), 400
        
        if not model:
            return jsonify({"error": "Model not loaded"}), 500
        
        with torch.no_grad():
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]
            
            pred_idx = torch.argmax(probabilities).item()
            confidence = probabilities[pred_idx].item()
            
            # Get scores for all classes
            scores = {
                id2label[i]: float(probabilities[i].item())
                for i in range(len(classes))
            }
            
            return jsonify({
                "predicted_class": id2label[pred_idx],
                "confidence": float(confidence),
                "all_scores": scores,
                "classes": classes
            })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/", methods=["GET"])
def index():
    """API documentation"""
    return jsonify({
        "name": "IT Service Ticket Classifier",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Classify a ticket",
            "/health": "GET - Health check"
        },
        "example": {
            "url": "/predict",
            "method": "POST",
            "body": {"text": "User cannot access shared drive"},
            "response": {
                "predicted_class": "Access",
                "confidence": 0.95,
                "all_scores": {"Access": 0.95, "Hardware": 0.03}
            }
        }
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
