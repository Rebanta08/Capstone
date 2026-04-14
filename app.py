"""
IT Service Ticket Classifier - Hugging Face Spaces
Serves ALBERT model predictions via Gradio API
"""

import gradio as gr
import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import json

# Load model and tokenizer
print("Loading ALBERT model...")
tokenizer = AlbertTokenizer.from_pretrained("./albert_ticket_classifier")
model = AlbertForSequenceClassification.from_pretrained("./albert_ticket_classifier", num_labels=8)

# Define classes
classes = ["Access", "Administrative rights", "HR Support", "Hardware", 
           "Internal Project", "Miscellaneous", "Purchase", "Storage"]
id2label = {i: cls for i, cls in enumerate(classes)}

device = torch.device("cpu")
model.to(device)
model.eval()

def classify_ticket(text):
    """
    Classify IT service ticket text
    Returns predicted class and confidence scores for all classes
    """
    if not text or len(text.strip()) == 0:
        return {"error": "Please enter ticket text"}
    
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
            id2label[i]: round(float(probabilities[i].item()), 4)
            for i in range(len(classes))
        }
        
        return {
            "predicted_class": id2label[pred_idx],
            "confidence": round(confidence, 4),
            "all_scores": scores
        }

# Gradio interface
def gradio_classify(ticket_text):
    """Gradio wrapper — returns JSON string for API consumers + display text"""
    result = classify_ticket(ticket_text)
    if "error" in result:
        return json.dumps(result)
    
    return json.dumps(result)

# Create Gradio interface
demo = gr.Interface(
    fn=gradio_classify,
    inputs=gr.Textbox(
        label="IT Service Ticket Description",
        placeholder="Describe your IT issue (e.g., 'Cannot access shared drives')",
        lines=4
    ),
    outputs=gr.JSON(label="Prediction"),
    title="IT Service Ticket Classifier",
    description="Enter your IT support ticket description and get instant classification into one of 8 categories using an ALBERT transformer model trained on 47,837 real tickets.",
    examples=[
        ["User cannot connect to VPN after password reset"],
        ["Printer is offline and not responding"],
        ["Need access to shared folder for project files"],
        ["Laptop monitor is broken, need replacement"],
        ["Employee requesting purchase approval for software license"],
        ["HR benefits question for new hires"],
        ["Server storage running low, need to expand capacity"],
    ]
)

if __name__ == "__main__":
    demo.launch()
