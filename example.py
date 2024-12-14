from src.core.pipeline import ImageGenerationPipeline
from src.config.lora_configs import get_lora_config, list_available_styles
from src.utils.image_utils import ImageProcessor

def main():
    # Initialize pipeline
    pipeline = ImageGenerationPipeline()
    
    # List available styles
    print("Available styles:")
    for style in list_available_styles():
        print(f"- {style}")
    
    # Generate image with specific style
    style = "Super Realism"
    lora_config = get_lora_config(style)
    
    # Generate image
    images = pipeline.generate_images(
        prompt=f"A beautiful landscape, {lora_config['trigger_word']}",
        num_images_per_prompt=1,
        lora_config=lora_config
    )
    
    # Process and save image
    processor = ImageProcessor()
    grid = processor.create_image_grid(images)
    grid.save("output.png")
    print("Image saved as output.png")

if __name__ == "__main__":
    main()
