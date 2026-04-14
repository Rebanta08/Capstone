# Deploy BERT Model to Hugging Face Spaces

## Quick Setup (5 minutes)

### Step 1: Create Hugging Face Account
1. Go to [huggingface.co](https://huggingface.co) and sign up
2. Create an **Access Token** (Settings → Access Tokens → New token)
3. Save it - you'll need it below

### Step 2: Create New Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → Create new Space
2. **Space name:** `ticket-classifier` (or your choice)
3. **Owner:** Your username
4. **License:** OpenRAIL-M
5. **Space SDK:** Select "Gradio"
6. Click "Create space"

### Step 3: Upload Model Files

In your Spaces repository (on HF), add these files:

**1. Copy `app.py`** (Gradio interface):
```bash
# From your Capstone repo
cp app.py YOUR_HF_SPACE_REPO/
```

**2. Copy model files**:
```bash
# Copy the bert_ticket_classifier directory
cp -r bert_ticket_classifier YOUR_HF_SPACE_REPO/
```

**3. Copy requirements.txt**:
```bash
cp requirements.txt YOUR_HF_SPACE_REPO/
```

### Step 4: Git Push to Hugging Face Spaces

```bash
cd YOUR_HF_SPACE_REPO
git add .
git commit -m "Add BERT ticket classifier"
git push
```

The Space will automatically build and deploy! ✅

---

## Your Space URL

Once deployed, your Space will be at:
```
https://[username]-ticket-classifier.hf.space
```

Example: `https://rebanta08-ticket-classifier.hf.space`

---

## Update Website with Your Space URL

In `index.html`, update line with your actual Space URL:

```javascript
const HF_API_URL = "https://YOUR_USERNAME-ticket-classifier.hf.space/api/predict/";
```

Replace `YOUR_USERNAME` with your Hugging Face username.

---

## Test the API

You can test directly in the browser:
```javascript
fetch("https://your-space-url.hf.space/api/predict/", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ data: ["Cannot access VPN"] })
})
.then(r => r.json())
.then(d => console.log(d));
```

---

## What Happens

1. User enters ticket text on website
2. Website calls your HF Spaces API
3. BERT model processes it
4. Returns predicted class + confidence scores
5. Website displays results

---

## Cost

**FREE!** 🎉
- Hugging Face Spaces has a free tier
- Spaces with low traffic get free GPU (runs on CPU)
- No credit card needed

---

## File Structure

```
Your HF Space:
├── app.py                      # Gradio interface
├── requirements.txt            # Dependencies
└── bert_ticket_classifier/     # Your trained model
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── metadata.json
```

---

## Troubleshooting

**"Model not loading" error:**
- Check that `bert_ticket_classifier/` directory is pushed
- Verify file names match exactly

**"API timeout" on website:**
- Cold start takes ~30 seconds on HF Spaces
- Add loading indicator feedback

**Wrong predictions:**
- Model trained on 8 IT ticket classes
- Check dataset matches your use case

---

## Next Steps

1. Deploy to HF Spaces (follow steps above)
2. Get your Space URL
3. Update `index.html` with URL
4. Push updated website to GitHub
5. Your website now uses real BERT model! ✅
