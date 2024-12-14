# LoRA Image Generation Pipeline

A powerful image generation pipeline using LoRA (Low-Rank Adaptation) models with support for multiple styles and configurations.

## Features

- Multiple LoRA model support
- Real-time image generation
- Memory-efficient processing
- Customizable generation parameters
- Support for various artistic styles

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/loraorwhat.git
cd loraorwhat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from src.core.pipeline import FluxPipeline

# Initialize pipeline
pipeline = FluxPipeline()

# Generate images
images = pipeline.generate(
    prompt="your prompt here",
    style="Super Realism",
    num_images=1
)
```

## Configuration

The system supports multiple LoRA models and styles. Check `src/config/lora_configs.py` for available styles.

## License

MIT License
