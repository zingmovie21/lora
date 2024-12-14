"""
Configuration file containing all available LoRA models and their settings.
"""

LORA_CONFIGS = [
    # Super-Realism
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Super-Realism-LoRA/resolve/main/images/1.png",
        "title": "Super Realism",
        "repo": "strangerzonehf/Flux-Super-Realism-LoRA",
        "weights": "super-realism.safetensors",
        "trigger_word": "Super Realism"            
    },
    # Dalle-Mix
    {
        "image": "https://huggingface.co/prithivMLmods/Flux-Dalle-Mix-LoRA/resolve/main/images/D3.png",
        "title": "Dalle Mix",
        "repo": "prithivMLmods/Flux-Dalle-Mix-LoRA",
        "weights": "dalle-mix.safetensors",
        "trigger_word": "dalle-mix"           
    },
    # Add more LoRA configurations here...
]

def get_lora_config(title: str) -> dict:
    """
    Get LoRA configuration by title.
    
    Args:
        title: Title of the LoRA configuration
    
    Returns:
        Dictionary containing LoRA configuration
    
    Raises:
        ValueError: If title not found
    """
    for config in LORA_CONFIGS:
        if config["title"].lower() == title.lower():
            return config
    raise ValueError(f"LoRA configuration '{title}' not found")

def list_available_styles() -> list:
    """
    Get list of available style titles.
    
    Returns:
        List of style titles
    """
    return [config["title"] for config in LORA_CONFIGS]

def get_style_info(title: str) -> dict:
    """
    Get detailed information about a style.
    
    Args:
        title: Title of the style
    
    Returns:
        Dictionary containing style information
    """
    config = get_lora_config(title)
    return {
        "title": config["title"],
        "preview_image": config["image"],
        "trigger_word": config["trigger_word"]
    }
