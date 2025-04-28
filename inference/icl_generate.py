import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from accelerate import PartialState
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import re
import torch
from pydantic import BaseModel, Field

# Clean up GPU memory
gc.collect()
torch.cuda.empty_cache()


class GenerationConfig(BaseModel):
    """Configuration for text generation."""
    max_length: int = Field(
        default=2048,
        description="Maximum length of generated text"
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter"
    )
    top_k: int = Field(
        default=0,
        ge=0,
        description="Top-k sampling parameter"
    )


def format_inference_prompt(instruction: str) -> str:
    """Format the instruction into a complete prompt for the model.
    
    Args:
        instruction (str): The user's instruction for layout generation
        
    Returns:
        str: Formatted prompt with context and examples
    """
    return f"""
    You are a specialized layout generation assistant for the `pdf-frame` framework — a declarative system for building PDF or canvas-based graphical templates.

    Important Constraints:
    - You must ONLY use the official `pdf-frame` tags listed below.
    - DO NOT use raw <svg>, <g>, <i-arc>, <path> (outside of <i-path>), <clipPath>, <defs>, or any pure SVG elements directly.
    - All templates must use pdf-frame primitives only.
    - Animations must use <i-animate> or <i-animatePath> inside allowed primitives

    Below is a reference of `pdf-frame` primitives, their purpose, and key attributes:

    - <i-rect>: Rectangle  
      `x`, `y`, `width`, `height`, `fill`, `stroke`, `rx`, `ry`

    - <i-circle>: Circle  
      `cx`, `cy`, `r`, `fill`, `stroke`

    - <i-text>: Text element  
      `x`, `y`, `text`, `font-size`, `font-weight`, `fill`, `font-family`

    - <i-path>: Path based on SVG `d` commands  
      `d`, `stroke`, `fill`
    
    - `<i-polygon>`: Draws polygons with multiple sides.  
      `:points` (list of `{{x, y}}` points), `:style` for stroke/fill.

    - `<i-polyline>`: Draws polylines (connected lines).  
      `:points` (list of `{{x, y}}` points), `:style` for stroke.

    - `<i-ellipse>`: Draws ellipses.  
      `cx`, `cy`, `rx`, `ry`, `:style` for fill.

    - <i-group>: Container for grouping elements  
      `:transform` (e.g., {{ translate: [x, y], scale: [sx, sy] }}")

    - <i-animate>: Animation for a primitive element (must be nested)  
      `:to`, `ease`, `:duration`, `loop`, `direction`, `:delay`

    - <i-animatePath>: Animates `<i-path>` `d` attribute  
      `:from`, `:to`, `:duration`, `ease`, `loop`, `direction`

    - <i-image>: Embeds an image  
      `x`, `y`, `width`, `height`, `src`

    - <i-linearGradient>, <i-radialGradient>: For gradient fills

    All elements must be nested inside a `<pdf-frame>` component:
    <pdf-frame type="canvas" width="600" height="400">
    <!-- elements go here -->
    </pdf-frame>

    Below are a few examples:
    ### Instruction:
    Generate a bar chart with labels A and B

    ### Response:
    <template>
    <pdf-frame type="canvas" width="600" height="400">
        <i-group :transform="{{ translate: [50, 60] }}'>
        <i-rect v-for="(d, i) in data" :x="i * 60" :y="200 - d.value" width="40" :height="d.value" fill="teal"/>
        <i-text v-for="(d, i) in data" :x="i * 60 + 10" y="220" :text="d.label" font-size="12"/>
        </i-group>
    </pdf-frame>
    </template>

    <script setup>
    const data = [ {{ label: 'A', value: 80 }}, {{ label: 'B', value: 120 }} ]
    </script>
    ### End

    ### Instruction:
    {instruction}

    ### Response:""".strip()


class ICLGenerator:
    """In-Context Learning Generator for PDF frame templates."""
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance of the generator exists."""
        if cls._instance is None:
            cls._instance = super(ICLGenerator, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: str):
        """Initialize the generator.
        
        Args:
            model_path (str): Path to the model
        """
        if ICLGenerator._initialized:
            return
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        ICLGenerator._initialized = True

    def load_model(self):
        """Load the model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
            ),
            device_map={"": PartialState().process_index}
        )

    def generate_inference(self, prompt: str, config: GenerationConfig) -> str:
        """Generate a PDF frame template based on the prompt.
        
        Args:
            prompt (str): The instruction prompt
            config (GenerationConfig): Generation configuration
            
        Returns:
            str: Generated template
            
        Raises:
            ValueError: If model is not loaded
            RuntimeError: If generation fails
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")
        
        if config is None:
            config = GenerationConfig()
        
        try:
            icl_prompt = format_inference_prompt(prompt)
            inputs = self.tokenizer(icl_prompt, return_tensors="pt")
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
                return self.extract_response(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        except Exception as e:
            raise RuntimeError(f"Error during generation: {str(e)}")
    
    def extract_response(text: str) -> str:
        matches = list(re.finditer(r"### Response:\s*", text))
        if not matches:
            return ""

        last_response_start = matches[-1].end()

        tail = text[last_response_start:]

        end_script = tail.find("</script>")
        end_template = tail.find("</template>")

        if end_script != -1:
            end_pos = end_script + len("</script>")
        elif end_template != -1:
            end_pos = end_template + len("</template>")
        else:
            end_pos = len(tail)

        return tail[:end_pos].strip()

def main(prompt: str):
    """Example usage of the ICLGenerator.
    
    Args:
        prompt (str): The instruction prompt
    """
    generator_instance = ICLGenerator(model_path="bigcode/starcoder2-7b")
    generator_instance.load_model()

    config = GenerationConfig(
        max_length=1024,
        temperature=0.3,
        top_p=0.9,
        top_k=0
    )

    inference = generator_instance.generate_inference(prompt, config)
    print(inference)


if __name__ == "__main__":
    main("Generate a bar chart with labels A and B")
