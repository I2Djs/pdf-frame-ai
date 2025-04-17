# PDF-Frame AI

A powerful AI-powered PDF frame template generator that uses fine-tuned language models to create optimized PDF frame templates based on natural language instructions.

## Features

- Generate PDF frame templates from natural language instructions
- Fine-tuned on StarCoder2-7B model
- GPU-accelerated inference
- REST API interface via FastAPI
- Support for charts, animations, and D3-based computations

## Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU with at least 16GB VRAM
- NVIDIA drivers and CUDA toolkit installed
- Hugging Face account and access token

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pdf-frame-ai.git
cd pdf-frame-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training (Fine-tuning)

1. Login to Hugging Face:
```bash
huggingface-cli login
```
You'll need to provide your Hugging Face access token. If you don't have one, you can create it at https://huggingface.co/settings/tokens

2. Start the training process:
```bash
python finetune/train.py
```

The training will:
- Download the base model from Hugging Face
- Download the training dataset from Hugging Face
- Fine-tune the model on the dataset
- Save the LoRA weights locally in the `./snaps/starcoder-pdf-frame-v2` directory

## Inference (Generation)

1. Start the FastAPI server:
```bash
uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

The server will:
- Load the base model from Hugging Face
- Load the local LoRA weights from `./snaps/starcoder-pdf-frame-v2`
- Start the inference server

### Command Line Interface

```python
from inference.generate import generator

# Initialize the generator
generator.load_model()

# Generate a template
result = generator.generate(
    "Generate a pdf-frame template for a resume with a modern design and sections for education, experience, and skills."
)
print(result)
```

### API Usage

```bash
# Generate a template
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Generate a pdf-frame template for a resume"}'
```

## API Documentation

Once the server is running, visit `http://localhost:8000/docs` for interactive API documentation.

### Endpoints

- `POST /generate`: Generate a PDF frame template
  - Request body:
    ```json
    {
        "prompt": "string"
    }
    ```
  - Response:
    ```json
    {
        "generated_text": "string"
    }
    ```

- `GET /health`: Check server health
  - Response:
    ```json
    {
        "status": "healthy",
        "model_loaded": true
    }
    ```

## Project Structure

```
pdf-frame-ai/
├── inference/
│   ├── generate.py    # Core generation logic
│   └── api.py         # FastAPI server
├── finetune/
│   └── train.py       # Model fine-tuning code
├── snaps/            # Directory for saved LoRA weights
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [StarCoder2](https://huggingface.co/bigcode/starcoder2-7b) for the base model
- [Hugging Face Transformers](https://huggingface.co/transformers/) for the model framework
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
