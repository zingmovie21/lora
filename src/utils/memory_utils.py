import torch
import gc
from typing import Optional

class MemoryManager:
    """Utility class for managing GPU memory."""
    
    @staticmethod
    def clear_cache():
        """Clear CUDA cache and garbage collect."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def get_memory_stats() -> dict:
        """
        Get current GPU memory statistics.
        
        Returns:
            Dictionary containing memory statistics
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
            
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated()
        }

    @staticmethod
    def estimate_batch_size(
        height: int,
        width: int,
        target_memory_usage: float = 0.7
    ) -> int:
        """
        Estimate optimal batch size based on available memory.
        
        Args:
            height: Image height
            width: Image width
            target_memory_usage: Target memory usage ratio (0.0 to 1.0)
        
        Returns:
            Estimated optimal batch size
        """
        if not torch.cuda.is_available():
            return 1

        total_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = total_memory * target_memory_usage
        
        # Rough estimation of memory per image
        pixels = height * width
        estimated_memory_per_image = pixels * 4 * 4  # Assuming float32 and 4 channels
        
        batch_size = max(1, int(available_memory / estimated_memory_per_image))
        return batch_size

class LatentMemoryPool:
    """Memory pool for managing latent tensors."""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.pool = []

    def add(self, tensor: torch.Tensor):
        """
        Add tensor to pool.
        
        Args:
            tensor: Tensor to add
        """
        if len(self.pool) >= self.max_size:
            self.pool.pop(0)
        self.pool.append(tensor)

    def get(self, index: int) -> Optional[torch.Tensor]:
        """
        Get tensor from pool.
        
        Args:
            index: Index of tensor to retrieve
        
        Returns:
            Tensor if found, None otherwise
        """
        if 0 <= index < len(self.pool):
            return self.pool[index]
        return None

    def clear(self):
        """Clear the pool."""
        self.pool = []
        MemoryManager.clear_cache()
