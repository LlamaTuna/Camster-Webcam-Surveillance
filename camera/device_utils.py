"""
Device utilities for PyTorch model inference.
Provides centralized device management with CUDA/CPU fallback.
"""
import torch
import logging

logger = logging.getLogger(__name__)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device for PyTorch inference.
    
    Args:
        prefer_cuda: If True, use CUDA when available. If False, always use CPU.
    
    Returns:
        torch.device: The selected device (cuda or cpu).
    """
    if prefer_cuda and torch.cuda.is_available():
        # Check if the GPU is actually compatible with this PyTorch build
        try:
            # Try a simple CUDA operation to verify compatibility
            test_tensor = torch.zeros(1, device='cuda')
            del test_tensor
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e:
            # GPU exists but isn't compatible (e.g., RTX 50 series with older PyTorch)
            logger.warning(f"CUDA available but incompatible: {e}")
            logger.info("Falling back to CPU due to GPU incompatibility")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        if prefer_cuda:
            logger.info("CUDA not available, falling back to CPU")
        else:
            logger.info("Using CPU (CUDA disabled by preference)")
    return device


def get_device_info() -> dict:
    """
    Get information about available compute devices.
    
    Returns:
        dict: Device information including CUDA availability and details.
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'device_name': None,
        'memory_allocated': None,
        'memory_reserved': None,
    }
    
    if torch.cuda.is_available():
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name(0)
        info['memory_allocated'] = torch.cuda.memory_allocated(0)
        info['memory_reserved'] = torch.cuda.memory_reserved(0)
    
    return info


def clear_gpu_memory():
    """Clear GPU memory cache if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU memory cache cleared")
