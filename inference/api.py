from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from .generate import PDFFrameGenerator, GenerationConfig
import torch

# Initialize FastAPI app
app = FastAPI(
    title="PDF Frame Generator API",
    description="API for generating PDF frame templates using fine-tuned language models",
    version="1.0.0"
)

# Initialize the generator
generator = PDFFrameGenerator(
    model_path="bigcode/starcoder2-7b",
    lora_path="/lora-models/starcoder-model-lora"
)

# Load the model when the API starts
@app.on_event("startup")
async def startup_event():
    try:
        generator.load_model()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )

# Unload the model and clear GPU resources when the API shuts down
@app.on_event("shutdown")
async def shutdown_event():
    try:
        # Clear model from memory
        if generator.model is not None:
            generator.model = None
            generator.tokenizer = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Error during shutdown: {str(e)}")

class GenerateRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 1024
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 0

class GenerateResponse(BaseModel):
    template: str

@app.post("/generate", response_model=GenerateResponse)
async def generate_template(request: GenerateRequest):
    """
    Generate a PDF frame template based on the provided prompt.
    
    Args:
        request (GenerateRequest): The request containing the prompt and optional generation parameters
        
    Returns:
        GenerateResponse: The generated PDF frame template
        
    Raises:
        HTTPException: If generation fails
    """
    try:
        config = GenerationConfig(
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        
        template = generator.generate(request.prompt, config)
        return GenerateResponse(template=template)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running and the model is loaded.
    
    Returns:
        dict: Status information
    """
    return {
        "status": "healthy",
        "model_loaded": generator.model is not None,
        "device": str(generator.device) if generator.device else None
    } 