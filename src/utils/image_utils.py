import torch
from PIL import Image
import numpy as np
from typing import Union, List, Tuple

class ImageProcessor:
    """Utility class for image processing operations."""
    
    @staticmethod
    def prepare_image(
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Prepare image for model input.
        
        Args:
            image: Input image
            height: Target height
            width: Target width
        
        Returns:
            Processed image tensor
        """
        if isinstance(image, Image.Image):
            image = image.resize((width, height), Image.LANCZOS)
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image = image.resize((width, height), Image.LANCZOS)
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.unsqueeze(0)
            image = torch.nn.functional.interpolate(
                image,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )
        
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1).unsqueeze(0)
        
        # Normalize
        image = image / 127.5 - 1.0
        return image

    @staticmethod
    def postprocess_image(
        image: torch.Tensor,
        output_type: str = "pil"
    ) -> Union[Image.Image, np.ndarray]:
        """
        Post-process generated image.
        
        Args:
            image: Input tensor
            output_type: Type of output ('pil' or 'numpy')
        
        Returns:
            Processed image
        """
        # Denormalize
        image = (image + 1.0) * 127.5
        image = image.clamp(0, 255).numpy().astype(np.uint8)
        
        # Reshape if needed
        if image.ndim == 4:
            image = image[0]
        image = np.transpose(image, (1, 2, 0))
        
        if output_type == "pil":
            return Image.fromarray(image)
        return image

    @staticmethod
    def create_image_grid(
        images: List[Union[Image.Image, np.ndarray]],
        rows: int = None,
        cols: int = None
    ) -> Image.Image:
        """
        Create a grid of images.
        
        Args:
            images: List of images
            rows: Number of rows (optional)
            cols: Number of columns (optional)
        
        Returns:
            Grid image
        """
        if rows is None and cols is None:
            cols = int(np.ceil(np.sqrt(len(images))))
            rows = int(np.ceil(len(images) / cols))
        elif rows is None:
            rows = int(np.ceil(len(images) / cols))
        elif cols is None:
            cols = int(np.ceil(len(images) / rows))
            
        # Convert all images to PIL
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)
                
        # Get max dimensions
        max_width = max(img.width for img in pil_images)
        max_height = max(img.height for img in pil_images)
        
        # Create grid
        grid = Image.new('RGB', (max_width * cols, max_height * rows))
        
        for idx, img in enumerate(pil_images):
            row = idx // cols
            col = idx % cols
            grid.paste(img, (col * max_width, row * max_height))
            
        return grid

    @staticmethod
    def resize_image(
        image: Union[Image.Image, np.ndarray],
        max_size: int = 1024,
        maintain_aspect: bool = True
    ) -> Union[Image.Image, np.ndarray]:
        """
        Resize image while maintaining aspect ratio if requested.
        
        Args:
            image: Input image
            max_size: Maximum size of the longest dimension
            maintain_aspect: Whether to maintain aspect ratio
        
        Returns:
            Resized image
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
            
        width, height = pil_image.size
        
        if maintain_aspect:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
        else:
            new_width = new_height = max_size
            
        resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        if isinstance(image, np.ndarray):
            return np.array(resized_image)
        return resized_image
