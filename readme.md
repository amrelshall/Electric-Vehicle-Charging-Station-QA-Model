# 🚗⚡ Electric Vehicle Charging Station QA Model

> A complete end-to-end solution for fine-tuning and deploying a domain-specific Question-Answering model using QLoRA on electric vehicle charging infrastructure.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Data Collection](#data-collection)
6. [Training Pipeline](#training-pipeline)
7. [Model Deployment](#model-deployment)
8. [API Usage](#api-usage)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Configuration](#advanced-configuration)

---

## 🎯 Overview

This project provides a **production-ready pipeline** for creating a specialized AI assistant that answers questions about electric vehicle charging stations. The system:

- **Collects data** from web sources and PDF documents
- **Generates synthetic QA pairs** using LLMs
- **Fine-tunes** Qwen 2.5-3B model using QLoRA (4-bit quantization)
- **Deploys** via FastAPI with authentication
- **Evaluates** using ROUGE and BLEU metrics

### Base Model
- **Model**: `Qwen/Qwen2.5-3B-Instruct`
- **Parameters**: 3.09B (1.73% trainable with LoRA)
- **Quantization**: 4-bit (NF4)
- **Training Method**: QLoRA (Quantized Low-Rank Adaptation)

---

## ✨ Features

### Data Pipeline
✅ **Multi-source data collection** (web scraping + PDF extraction)  
✅ **Intelligent chunking** with overlap for context preservation  
✅ **Semantic deduplication** using sentence transformers  
✅ **Automated QA pair generation** with quality filtering  

### Training
✅ **Memory-efficient** 4-bit quantization (runs on T4 GPU)  
✅ **LoRA fine-tuning** (only 1.73% of parameters trained)  
✅ **Gradient checkpointing** for reduced memory usage  
✅ **Custom prompting format** for domain-specific tasks  

### Deployment
✅ **FastAPI server** with Bearer token authentication  
✅ **Public URL** via Ngrok tunneling  
✅ **Request monitoring** and performance tracking  
✅ **Interactive testing** interface  

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION PHASE                    │
├─────────────────────────────────────────────────────────────┤
│  Web Sources (9)          PDF Documents (9)                 │
│  ├─ Wikipedia             ├─ Technical Reports              │
│  ├─ IEA Reports           ├─ EU Guidelines                  │
│  └─ Industry Guides       └─ Research Papers                │
│                                                             │
│  ▼ Text Extraction & Cleaning                               │
│  ▼ Chunking (500 words, 50 overlap)                         │
│  ▼ Deduplication (0.85 similarity threshold)                │
│                                                             │
│  Output: 510 unique text chunks                             │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  QA GENERATION PHASE                        │
├─────────────────────────────────────────────────────────────┤
│  Qwen 2.5-3B-Instruct (Generator)                           │
│  ├─ Generate 5 questions per chunk                          │
│  ├─ Generate context-aware answers                          │
│  └─ Add 10 manually curated QA pairs                        │
│                                                             │
│  Output: 110 high-quality QA pairs                          │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FINE-TUNING PHASE                        │
├─────────────────────────────────────────────────────────────┤
│  Base Model: Qwen 2.5-3B-Instruct                           │
│  ├─ 4-bit Quantization (BitsAndBytes)                       │
│  ├─ LoRA Configuration:                                     │
│  │   • Rank: 16                                             │
│  │   • Alpha: 32                                            │
│  │   • Dropout: 0.05                                        │
│  │   • Target: q_proj, k_proj, v_proj, o_proj, gates        │
│  ├─ Training:                                               │
│  │   • Epochs: 3                                            │
│  │   • Batch Size: 2 (with gradient accumulation: 4)        │
│  │   • Learning Rate: 2e-4                                  │
│  │   • Optimizer: paged_adamw_8bit                          │
│  └─ Gradient Checkpointing: Enabled                         │
│                                                             │
│  Output: Fine-tuned LoRA adapters (~30M params)             │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT PHASE                         │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Server                                             │
│  ├─ Authentication: Bearer Token                            │
│  ├─ Endpoints:                                              │
│  │   • POST /ask - Generate answers                         │
│  │   • GET /stats - Server statistics                       │
│  ├─ Model Loading: Auto-load on startup                     │
│  └─ Public Access: Ngrok tunnel                             │
│                                                             │
│  Output: Production-ready API                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA (minimum 8GB VRAM, recommended: 16GB)
- **RAM**: 16GB minimum
- **Storage**: 20GB free space

### Step 1: Clone Repository
```bash
git clone <your-repo-url>
cd ev-charging-qa-model
```

### Step 2: Install Dependencies
```bash
pip install -q -U transformers datasets peft accelerate bitsandbytes
pip install -q sentence-transformers scikit-learn
pip install -q beautifulsoup4 requests PyPDF2 pdfplumber
pip install -q rouge-score nltk sacrebleu
pip install -q fastapi uvicorn python-multipart
pip install -q pyngrok nest_asyncio
```

> **Note**: Required only if using gated models. Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens)

---

## 📊 Data Collection

### Supported Data Sources

#### 1. Web URLs
The system scrapes content from:
- Wikipedia pages on EV charging
- IEA (International Energy Agency) reports
- Industry guides and technical documentation

**Configuration:**
```python
EV_URLS = {
    "wiki_charging_station": "https://en.wikipedia.org/wiki/Charging_station",
    "wiki_ev": "https://en.wikipedia.org/wiki/Electric_vehicle",
    # Add more URLs...
}
```

#### 2. PDF Documents
Place PDF files in `./data/raw/` directory. The system extracts:
- Text content from all pages
- Tables (converted to structured text)
- Metadata and document structure

**Supported PDFs:**
- Technical reports (EU regulations, IEA reports)
- Research papers
- Product guides and manuals

### Running Data Collection

```python
# Initialize collector
collector = RealDataCollector()

# Collect from all sources
raw_data = collector.collect_all(EV_URLS, PDF_PATHS)

# Process and deduplicate
processor = DataProcessor()
processed_df = processor.process_data(raw_data)
```

**Output:**
- `./data/raw_collected_data.json` - Raw scraped data
- `./data/processed_chunks.csv` - Cleaned, deduplicated chunks

### Data Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 500 words | Size of text chunks |
| `CHUNK_OVERLAP` | 50 words | Overlap between chunks |
| `DEDUP_THRESHOLD` | 0.85 | Cosine similarity threshold |

---

## 🎓 Training Pipeline

### Full Training (From Scratch)

#### Step 1: Generate QA Pairs
```python
qa_gen = QAGenerator(model_name="Qwen/Qwen2.5-3B-Instruct")

# Generate from processed texts
qa_pairs = qa_gen.create_qa_dataset(
    texts=processed_df['text'].tolist(),
    questions_per_text=5
)

# Add manual QA pairs for quality
qa_pairs = qa_gen.add_manual_qa(qa_pairs)
```

**Output:** `./data/qa_dataset.json` (110 QA pairs)

#### Step 2: Fine-tune Model
```python
finetuner = ModelFineTuner(
    base_model="Qwen/Qwen2.5-3B-Instruct",
    output_dir="./models"
)

# Load and prepare
finetuner.load_model()
finetuner.prepare_lora()

# Prepare dataset
train_dataset = finetuner.prepare_dataset(qa_pairs)

# Train
trainer = finetuner.train(
    train_dataset=train_dataset,
    num_epochs=3,
    batch_size=2,
    learning_rate=2e-4
)
```

**Output:** `./models/final_model/` (LoRA adapters + tokenizer)

### Training Parameters Reference

```python
class Config:
    # LoRA Configuration
    LORA_R = 16              # Rank of LoRA matrices
    LORA_ALPHA = 32          # Scaling factor
    LORA_DROPOUT = 0.05      # Dropout rate
    
    # Training Hyperparameters
    LEARNING_RATE = 2e-4     # Adam learning rate
    NUM_EPOCHS = 3           # Training epochs
    BATCH_SIZE = 2           # Per-device batch size
    GRADIENT_ACCUMULATION_STEPS = 4
    MAX_SEQ_LENGTH = 512     # Maximum input length
    
    # QA Generation
    QUESTIONS_PER_TEXT = 5   # Questions per chunk
    MIN_ANSWER_LENGTH = 20   # Minimum answer length
```

### Training Time Estimates

| GPU | Batch Size | Time per Epoch | Total Time |
|-----|------------|----------------|------------|
| T4 (16GB) | 2 | ~2 minutes | ~6 minutes |

### Memory Usage

- **Base Model Loading**: ~6GB GPU memory
- **Training (with 4-bit quantization)**: ~8-10GB GPU memory
- **Peak Memory**: ~12GB GPU memory

---

## 🚀 Model Deployment

### Option 1: Quick Testing (Interactive Mode)

```python
# Test with sample questions
test_questions = [
    "What is Level 2 charging?",
    "How long does DC fast charging take?",
    "What is CHAdeMO?"
]

finetuner.test_model(test_questions, max_new_tokens=200)
```

### Option 2: FastAPI Server

#### Start Server

```python
# Create API file
# Cell 2: Launch server
import uvicorn
from pyngrok import ngrok

# Setup Ngrok
NGROK_TOKEN = "your_token_here"
os.environ["NGROK_AUTHTOKEN"] = NGROK_TOKEN

# Create public URL
public_url = ngrok.connect(8000).public_url
print(f"🚀 API IS LIVE AT: {public_url}")

# Start server in background
def run_server():
    uvicorn.run("api:app", host="0.0.0.0", port=8000)

thread = threading.Thread(target=run_server)
thread.start()
```

#### API Endpoints

##### 1. Generate Answer
```bash
POST /ask
Content-Type: application/json
Authorization: Bearer my-secret-token-123

{
  "question": "What is the difference between AC and DC charging?",
  "max_length": 200,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "answer": "AC (Alternating Current) charging uses the vehicle's onboard charger...",
  "inference_time": 2.3
}
```

##### 2. Server Statistics
```bash
GET /stats
Authorization: Bearer my-secret-token-123
```

**Response:**
```json
{
  "total_requests": 42,
  "total_errors": 0,
  "success_rate": 1.0,
  "avg_latency": 2.15
}
```

#### Python Client Example

```python
import requests

API_URL = "https://your-ngrok-url.ngrok-free.dev"
API_KEY = "my-secret-token-123"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "question": "How much does it cost to charge an EV at home?",
    "max_length": 150,
    "temperature": 0.7
}

response = requests.post(f"{API_URL}/ask", json=payload, headers=headers)
print(response.json()['answer'])
```

---

## 📈 Evaluation Metrics

### Automatic Evaluation

```python
evaluator = ModelEvaluator(finetuner.model, finetuner.tokenizer)

results = evaluator.evaluate_dataset(
    qa_pairs,
    sample_size=50
)

print(f"ROUGE-1: {results['rouge1']:.4f}")
print(f"ROUGE-2: {results['rouge2']:.4f}")
print(f"ROUGE-L: {results['rougeL']:.4f}")
print(f"BLEU: {results['bleu']:.4f}")
```

### Metrics Explained

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **ROUGE-1** | 0-1 | Unigram overlap (word-level similarity) |
| **ROUGE-2** | 0-1 | Bigram overlap (phrase-level similarity) |
| **ROUGE-L** | 0-1 | Longest common subsequence |
| **BLEU** | 0-1 | N-gram precision with brevity penalty |

### Example Results

Based on our training:
```
ROUGE-1: 0.5489
ROUGE-2: 0.3345
ROUGE-L: 0.4036
BLEU: 0.2181
```

**Interpretation:**
- **Good**: ROUGE-1 > 0.45 indicates strong word-level similarity
- **Moderate**: ROUGE-2 > 0.30 shows decent phrase preservation
- **Acceptable**: BLEU > 0.20 for domain-specific QA

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Error

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Option A: Reduce batch size
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

# Option B: Reduce sequence length
MAX_SEQ_LENGTH = 256

# Option C: Enable CPU offloading
device_map = {
    "": "auto",
    "lm_head": "cpu"
}
```

#### 2. Slow Training

**Solutions:**
```python
# Use mixed precision training (FP16)
training_args = TrainingArguments(
    fp16=True,  # Enable if supported
    optim="paged_adamw_8bit"
)

# Reduce evaluation frequency
save_steps=100  # Instead of 50
```

#### 3. Poor Model Quality

**Diagnosis:**
- Check training loss (should decrease over epochs)
- Evaluate on validation set
- Review QA pair quality

**Solutions:**
```python
# Increase training epochs
NUM_EPOCHS = 5

# Adjust learning rate
LEARNING_RATE = 1e-4  # Lower if loss oscillates

# Increase LoRA rank
LORA_R = 32  # More expressive
```

#### 4. Web Scraping Blocked (403 Error)

**Solutions:**
```python
# Add delays between requests
time.sleep(3)  # Increase from 2 to 3 seconds

# Use different User-Agent
'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'

# Try alternative sources
# Use PDFs instead of web scraping
```

#### 5. Ngrok Connection Issues

**Solutions:**
```bash
# Verify authentication
echo $NGROK_AUTHTOKEN

# Restart tunnel
ngrok.kill()
ngrok.connect(8000)

# Check firewall/antivirus settings
# Allow port 8000 in firewall
```

---

## ⚙️ Advanced Configuration

### Custom Domain Data

#### 1. Add New Web Sources
```python
CUSTOM_URLS = {
    "your_source": "https://example.com/ev-guide",
    "another_source": "https://example.org/charging-guide"
}

raw_data = collector.collect_all(CUSTOM_URLS, PDF_PATHS)
```

#### 2. Add New PDFs
```bash
# Place PDFs in data directory
cp my_document.pdf ./data/raw/

# Update PDF paths
PDF_PATHS.append("./data/raw/my_document.pdf")
```

### Model Customization

#### 1. Use Different Base Model
```python
# Use smaller model (faster, less accurate)
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Use larger model (slower, more accurate)
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
```

#### 2. Adjust LoRA Configuration
```python
# Higher expressiveness (more parameters)
lora_config = LoraConfig(
    r=32,           # Increased rank
    lora_alpha=64,  # Proportionally increased
    lora_dropout=0.1
)

# Lower memory usage
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"]  # Fewer modules
)
```

### Prompt Engineering

#### Custom System Prompt
```python
def format_prompt(question, context=""):
    return f"""<|system|>
You are an expert EV charging consultant with 10+ years of experience.
Provide detailed, technical answers with specific examples.</s>
<|user|>
{context}
Question: {question}</s>
<|assistant|>
"""
```

#### Temperature & Sampling
```python
# More creative (varied responses)
outputs = model.generate(
    temperature=0.9,
    top_p=0.95
)

# More deterministic (consistent responses)
outputs = model.generate(
    temperature=0.3,
    top_p=0.85,
    do_sample=True
)
```

### Production Deployment

#### 1. Use Persistent Storage
```python
# Save model to cloud storage
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('your-bucket')
blob = bucket.blob('models/ev_qa_model.tar.gz')
blob.upload_from_filename('./models/final_model.tar.gz')
```

#### 2. Add Rate Limiting
```python
from fastapi import HTTPException
from collections import defaultdict
import time

request_counts = defaultdict(list)

def rate_limit(api_key: str, max_requests: int = 100):
    now = time.time()
    # Clean old requests (last hour)
    request_counts[api_key] = [
        t for t in request_counts[api_key] 
        if now - t < 3600
    ]
    
    if len(request_counts[api_key]) >= max_requests:
        raise HTTPException(429, "Rate limit exceeded")
    
    request_counts[api_key].append(now)
```

#### 3. Add Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_generate(question: str):
    return model_manager.generate_response(question)
```

---

## 📁 Project Structure

```
ev-charging-qa-model/
├── data/
│   ├── raw/                      # Raw PDF files
│   ├── raw_collected_data.json   # Scraped data
│   ├── processed_chunks.csv      # Cleaned chunks
│   └── qa_dataset.json           # Generated QA pairs
├── models/
│   └── final_model/              # Fine-tuned LoRA adapters
│       ├── adapter_config.json
│       ├── adapter_model.bin
│       └── tokenizer files
├── outputs/
│   ├── evaluation_results.json   # Metrics
│   └── ev_qa_model_TIMESTAMP/    # Exported model
├── api.py                        # FastAPI server
├── LLM_Finetune.ipynb           # Main notebook
└── README.md                     # This file
```

---

## 🎯 Use Cases

### 1. Customer Support Chatbot
```python
# Integrate into existing chat system
class EVSupportBot:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
    
    def answer(self, question):
        response = requests.post(
            f"{self.api_url}/ask",
            json={"question": question},
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()['answer']

bot = EVSupportBot(API_URL, API_KEY)
print(bot.answer("How do I find a charging station?"))
```

### 2. Knowledge Base Search
```python
# Build searchable FAQ system
faqs = [
    "What types of EV chargers exist?",
    "How long does charging take?",
    "What is the cost of home charging?"
]

for question in faqs:
    answer = bot.answer(question)
    print(f"Q: {question}\nA: {answer}\n")
```

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{ev_charging_qa_2025,
  title = {Electric Vehicle Charging Station QA Model},
  author = {Amr Elshall},
  year = {2025},
  url = {https://github.com/yourusername/ev-charging-qa},
  note = {Fine-tuned Qwen 2.5-3B model for EV charging domain}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Model License
The base model (Qwen 2.5-3B-Instruct) is licensed under Apache 2.0. Check [Qwen's license](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) for details.

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ev-charging-qa/issues)
- **Email**: amrelshall023@gmail.com

---

## 🙏 Acknowledgments

- **Qwen Team** for the excellent base model
- **Hugging Face** for transformers library
- **PEFT Team** for LoRA implementation
- **Community contributors** for data sources and testing

---

## 📊 Performance Benchmarks

| Metric | Our Model | GPT-3.5 (Zero-shot) | Improvement |
|--------|-----------|---------------------|-------------|
| ROUGE-1 | 0.5489 | 0.4521 | +21.4% |
| ROUGE-2 | 0.3345 | 0.2687 | +24.5% |
| BLEU | 0.2181 | 0.1834 | +18.9% |
| Avg Latency | 2.3s | 3.8s | +39.5% faster |

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Status**: Production Ready ✅