import torch
from typing import Any, Dict, List, Optional, Union
from diffusers import (
    DiffusionPipeline,
    AutoencoderTiny,
    AutoencoderKL,
    FluxPipeline,
    FlowMatchEulerDiscreteScheduler
)

class ImageGenerationPipeline:
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model = self._initialize_model(model_id)
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        self._setup_pipeline()

    def _initialize_model(self, model_id: str) -> DiffusionPipeline:
        """Initialize the base model."""
        return DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

    def _setup_pipeline(self):
        """Setup additional pipeline components."""
        self.scheduler = FlowMatchEulerDiscreteScheduler()
        self.model.scheduler = self.scheduler

    @torch.inference_mode()
    def generate_images(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        lora_config: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        Generate images based on the provided prompt and configuration.
        
        Args:
            prompt: Main prompt for image generation
            prompt_2: Optional secondary prompt
            height: Output image height
            width: Output image width
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            num_images_per_prompt: Number of images to generate
            lora_config: LoRA model configuration to use
        
        Returns:
            List of generated images
        """
        # Apply LoRA if config provided
        if lora_config:
            self._apply_lora(lora_config)

        # Generate images
        images = []
        for img in self.model(
            prompt=prompt,
            prompt_2=prompt_2,
            height=height or self.model.config.sample_size * self.model.vae_scale_factor,
            width=width or self.model.config.sample_size * self.model.vae_scale_factor,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
        ):
            images.append(img)
            torch.cuda.empty_cache()

        return images

    def _apply_lora(self, lora_config: Dict[str, Any]):
        """Apply LoRA weights to the model."""
        if "weights" in lora_config:
            self.model.load_lora_weights(
                lora_config["repo"],
                weight_name=lora_config["weights"],
            )
        else:
            self.model.load_lora_weights(lora_config["repo"])

    def __del__(self):
        """Cleanup resources."""
        torch.cuda.empty_cache()
