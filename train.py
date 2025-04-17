import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from typing import Tuple, Optional
from datasets import Dataset, load_dataset
from trl import SFTTrainer
from accelerate import PartialState
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
import torch
import gc
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)


class PDFFrameTrainer:
    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        lora_output_path: str,
        output_dir: str = "./snaps/starcoder-pdf-frame-v2",
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 2e-5,
        warmup_steps: int = 20,
        save_steps: int = 500,
        logging_steps: int = 500,
        seed: int = 0
    ):
        """Initialize the PDFFrameTrainer with training configuration.
        
        Args:
            model_path (str): Path to the pre-trained model
            dataset_path (str): Path to the training dataset
            output_dir (str): Directory to save checkpoints
            batch_size (int): Training batch size
            gradient_accumulation_steps (int): Number of steps for gradient accumulation
            learning_rate (float): Learning rate for training
            warmup_steps (int): Number of warmup steps
            save_steps (int): Save checkpoint every n steps
            logging_steps (int): Log metrics every n steps
            seed (int): Random seed for reproducibility
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.lora_output_path = lora_output_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.seed = seed
        
        self._validate_paths()
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _validate_paths(self) -> None:
        """Validate that required paths exist."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path {self.model_path} does not exist")
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path {self.dataset_path} does not exist")
        
        os.makedirs(self.output_dir, exist_ok=True)

    def _validate_gpu(self) -> None:
        """Validate GPU availability and setup."""
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. Training requires GPU acceleration.")
        
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load and prepare the model for training.
        
        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer
        """
        gc.collect()
        torch.cuda.empty_cache()
        
        self._validate_gpu()
        
        print("Loading the tokenizer and model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map={"": PartialState().process_index}
        )

        # Prepare model for training
        model = prepare_model_for_kbit_training(base_model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=8,
            target_modules=[
                "q_proj",
                "o_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        model = model.to("cuda")
        
        model.print_trainable_parameters()
        self.model = model
        print("Model loaded successfully!")
        
        return model, self.tokenizer

    def format_prompt(self, example: dict) -> dict:
        """Format training examples into prompts.
        
        Args:
            example (dict): Training example containing instruction and output
            
        Returns:
            dict: Tokenized prompt with labels
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not initialized. Call load_model() first.")
            
        prompt = f"""
        You are a pdf-frame layout generation agent. Your job is to generate valid and optimized pdf-frame templates based on the given instruction. The output should strictly follow pdf-frame syntax, and may include chart logic, animation, or D3-based computation as needed.
        
        ### Instruction:
        {example['instruction']}
        \n
        ### Response:
        {example['output']}
        """.strip()
        
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=1024
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def load_dataset(self) -> Dataset:
        """Load and prepare the training dataset.
        
        Returns:
            Dataset: Prepared training dataset
        """
        dataset = load_dataset(
            "json",
            data_files=self.dataset_path
        )["train"]
        
        print(f"Loaded dataset with {len(dataset)} examples")
        return dataset

    def prepare_trainer(self) -> Trainer:
        """Prepare the trainer with model and dataset.
        
        Returns:
            Trainer: Configured trainer instance
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer not initialized. Call load_model() first.")
            
        dataset = self.load_dataset()
        tokenized_dataset = dataset.map(self.format_prompt)

        training_args = TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            learning_rate=self.learning_rate,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            bf16=True,
            logging_strategy="steps",
            dataloader_num_workers=4,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            output_dir=self.output_dir,
            optim="paged_adamw_8bit",
            seed=self.seed,
            run_name=f"train-starcoder2-3b",
            report_to="none"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        
        return self.trainer

    def train(self) -> None:
        """Start the training process."""
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call prepare_trainer() first.")
            
        print("Starting training...")
        self.trainer.train()
        print("Training completed!")
    def save_model(self) -> None:
        """Save the model and tokenizer."""
        if not self.model or not self.tokenizer or not self.trainer:
            raise RuntimeError("Model and tokenizer not initialized or trainer not initialized. Call load_model() and prepare_trainer() first.")
            
        self.model.save_pretrained(self.lora_output_path)
        self.tokenizer.save_pretrained(self.lora_output_path)
        self.trainer.save_model(self.lora_output_path)


def main():
    # Example usage
    trainer = PDFFrameTrainer(
        model_path="bigcode/starcoder2-7b",
        dataset_path="https://huggingface.co/datasets/nswamy14/pdf-frame-dataset-1/resolve/main/pdf_frame_dataset_large.jsonl",
        lora_output_path="/lora-models/starcoder-model-lora"
    )
    
    trainer.load_model()
    trainer.prepare_trainer()
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()