from pathlib import Path
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import gc
from peft import PeftModel
from accelerate import PartialState
from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    """Configuration for text generation."""
    max_length: int = Field(default=1024, description="Maximum length of generated text")
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation (0.0 to 1.0)"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter (0.0 to 1.0)"
    )
    top_k: int = Field(
        default=0,
        ge=0,
        description="Top-k sampling parameter (0 or greater)"
    )


class PDFFrameGenerator:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PDFFrameGenerator, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: str, lora_path: str):
        """Initialize the PDFFrameGenerator.
        
        Args:
            model_path (str): Path to the base model
            lora_path (str): Path to the LoRA weights
        """
        if PDFFrameGenerator._initialized:
            return
            
        self.model_path = model_path
        self.lora_path = lora_path
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: Optional[torch.device] = None
        PDFFrameGenerator._initialized = True
        
    def load_model(self) -> None:
        """Load the model and tokenizer.
        
        Raises:
            RuntimeError: If GPU is not available
            FileNotFoundError: If model or LoRA paths don't exist
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path {self.model_path} does not exist")
        if not os.path.exists(self.lora_path):
            raise FileNotFoundError(f"LoRA path {self.lora_path} does not exist")
            
        gc.collect()
        torch.cuda.empty_cache()

        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("GPU is required for inference")

        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.lora_path)
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map={"": PartialState().process_index}
            )
            
            self.model = PeftModel.from_pretrained(base_model, self.lora_path)
            self.model.eval()  # Set to evaluation mode
            
            print("Model loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def format_prompt(self, prompt: str) -> str:
        """Format the input prompt for the model.
        
        Args:
            prompt (str): The input instruction
            
        Returns:
            str: Formatted prompt
        """
        return f"""
        You are a pdf-frame layout generation agent. Your job is to generate valid and optimized pdf-frame templates based on the given instruction. The output should strictly follow pdf-frame syntax, and may include chart logic, animation, or D3-based computation as needed.
        
        ### Instruction:
        {prompt}
        \n
        ### Response:
        """.strip()
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Generate a PDF frame template based on the input prompt.
        
        Args:
            prompt (str): The instruction prompt
            config (Optional[GenerationConfig]): Generation configuration
            
        Returns:
            str: Generated PDF frame template
            
        Raises:
            RuntimeError: If model or tokenizer is not loaded
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Please call load_model() first.")
        
        if config is None:
            config = GenerationConfig()
        
        try:
            formatted_prompt = self.format_prompt(prompt)
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)

            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)    

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=config.max_length,
                    do_sample=True,
                    repetition_penalty=1.2,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            raise RuntimeError(f"Error during generation: {str(e)}")


# Create a global instance that can be imported and used across the application
generator = PDFFrameGenerator(
    model_path="bigcode/starcoder2-7b",
    lora_path="/lora-models/starcoder-model-lora"
)


def main():
    """Example usage of the PDFFrameGenerator."""
    # Use the global instance
    generator.load_model()
    
    config = GenerationConfig(
        max_length=1024,
        temperature=0.3,
        top_p=0.9,
        top_k=0
    )
    
    result = generator.generate(
        "Generate a pdf-frame template for a resume.",
        config=config
    )
    print(result)


if __name__ == "__main__":
    main()