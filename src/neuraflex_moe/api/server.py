"""FastAPI server for NeuralFlex-MoE"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import torch
import uvicorn


app = FastAPI(
    title="NeuralFlex-MoE API",
    description="API for NeuralFlex-MoE language model",
    version="0.1.0"
)

# Global model instance (loaded on startup)
model = None
inference_engine = None


class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    stream: Optional[bool] = False


class GenerationResponse(BaseModel):
    text: str
    confidence: Optional[float] = None
    reasoning_steps: Optional[List[str]] = None
    alternatives: Optional[List[str]] = None
    tokens_used: int


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, inference_engine
    
    try:
        from ..models import NeuralFlexMoE
        from ..config import MODEL_CONFIG
        from ..inference import OptimizedInference
        from transformers import AutoTokenizer
        
        print("Loading model...")
        model = NeuralFlexMoE(MODEL_CONFIG)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        
        inference_engine = OptimizedInference(model, tokenizer)
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "NeuralFlex-MoE API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "NeuralFlex-MoE",
        "model_loaded": model is not None
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate text from prompt"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = inference_engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        
        return GenerationResponse(
            text=result["text"],
            tokens_used=result["num_tokens"],
            confidence=None,
            reasoning_steps=None,
            alternatives=None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_with_uncertainty")
async def generate_with_uncertainty(request: GenerationRequest):
    """Generate with uncertainty awareness"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        from ..core import UncertaintyAwareGeneration
        from ..config import NOVEL_FEATURES_CONFIG
        
        uag = UncertaintyAwareGeneration(NOVEL_FEATURES_CONFIG["uncertainty_aware_generation"])
        
        # Tokenize
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        input_ids = tokenizer.encode(request.prompt, return_tensors="pt")
        
        # Generate with uncertainty
        result = uag.generate_with_confidence(
            model=model,
            input_ids=input_ids,
            max_length=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Decode
        generated_text = tokenizer.decode(result["tokens"][0], skip_special_tokens=True)
        
        return {
            "text": generated_text,
            "confidence": result["confidence"],
            "uncertainty_flag": result["uncertainty_flag"],
            "tokens_used": len(result["tokens"][0])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
