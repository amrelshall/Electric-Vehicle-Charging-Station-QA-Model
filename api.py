
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time
import logging
from datetime import datetime
import os
from collections import deque

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration (UPDATED FOR YOUR QWEN MODEL)
# ============================================================================

class ServerConfig:
    # تم التعديل ليشير إلى مسار النموذج الذي قمت بحفظه في الكود السابق
    MODEL_PATH = "./models/final_model" 
    
    # تم التعديل ليتوافق مع النموذج الأساسي الذي استخدمته في التدريب
    BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct" 
    
    API_KEY = "my-secret-token-123"  # مفتاح بسيط للتجربة
    MAX_LENGTH = 512
    TEMPERATURE = 0.7
    TOP_P = 0.9

config = ServerConfig()

# ============================================================================
# Security
# ============================================================================

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    if token != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return token

# ============================================================================
# Monitoring
# ============================================================================

class RequestMonitor:
    def __init__(self, max_size=1000):
        self.requests = deque(maxlen=max_size)
        self.total_requests = 0
        self.total_errors = 0
    
    def log_request(self, endpoint: str, latency: float, success: bool, error: str = None):
        self.total_requests += 1
        if not success:
            self.total_errors += 1
        
        self.requests.append({
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'latency': latency,
            'success': success,
            'error': error
        })
    
    def get_stats(self) -> Dict:
        recent_requests = list(self.requests)
        if not recent_requests:
            return {'total_requests': self.total_requests, 'total_errors': self.total_errors}
        
        latencies = [r['latency'] for r in recent_requests]
        success_rate = sum(1 for r in recent_requests if r['success']) / len(recent_requests)
        
        return {
            'total_requests': self.total_requests,
            'total_errors': self.total_errors,
            'success_rate': success_rate,
            'avg_latency': sum(latencies) / len(latencies)
        }

monitor = RequestMonitor()

# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self):
        try:
            logger.info(f"Loading base model: {config.BASE_MODEL}...")
            self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
            
            base_model = AutoModelForCausalLM.from_pretrained(
                config.BASE_MODEL,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            
            logger.info(f"Loading LoRA weights from: {config.MODEL_PATH}...")
            self.model = PeftModel.from_pretrained(base_model, config.MODEL_PATH)
            self.model.eval()
            
            logger.info("Model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def generate_response(self, question: str, max_length: int = None, temperature: float = None):
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # تنسيق البرومبت الخاص بـ Qwen Instruct
        prompt = f"""<|system|>
You are an expert assistant for electric vehicle charging stations. Answer strictly based on your fine-tuned knowledge.</s>
<|user|>
{question}</s>
<|assistant|>
"""
        
        start_time = time.time()
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length or config.MAX_LENGTH,
                    temperature=temperature or config.TEMPERATURE,
                    top_p=config.TOP_P,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # استخراج الرد فقط (ما بعد assistant)
            if "<|assistant|>" in full_response:
                answer = full_response.split("<|assistant|>")[-1].strip()
            else:
                # محاولة بديلة للتنظيف إذا فشل التقسيم
                answer = full_response.replace(prompt, "").strip()

            inference_time = time.time() - start_time
            return {'answer': answer, 'inference_time': inference_time}
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

model_manager = ModelManager()

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="EV Charging QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    max_length: Optional[int] = 200
    temperature: Optional[float] = 0.7

@app.on_event("startup")
async def startup_event():
    model_manager.load_model()

@app.get("/")
def root():
    return {"message": "EV Charging QA Model Online", "docs": "/docs"}

@app.post("/ask")
async def ask(request: QuestionRequest, token: str = Depends(verify_token)):
    start_time = time.time()
    try:
        result = model_manager.generate_response(
            request.question, request.max_length, request.temperature
        )
        monitor.log_request("/ask", time.time() - start_time, True)
        return result
    except Exception as e:
        monitor.log_request("/ask", time.time() - start_time, False, str(e))
        raise

@app.get("/stats")
async def stats(token: str = Depends(verify_token)):
    return monitor.get_stats()
