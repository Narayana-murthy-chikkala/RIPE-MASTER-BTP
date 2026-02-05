# Optimized VGG Hypercolumn Feature Extractor Initialization
# With model caching, memory optimization, and flexible configurations

from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class ModelCache:
    """Simple model weight cache for multiple instantiations."""
    
    _cache: Dict[str, Dict[str, torch.Tensor]] = {}
    
    @classmethod
    def get(cls, key: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get cached model weights."""
        return cls._cache.get(key)
    
    @classmethod
    def set(cls, key: str, weights: Dict[str, torch.Tensor]) -> None:
        """Cache model weights."""
        cls._cache[key] = weights
    
    @classmethod
    def clear(cls) -> None:
        """Clear entire cache."""
        cls._cache.clear()


def _validate_model_path(model_path: Path) -> Path:
    """Validate and resolve model path.
    
    Args:
        model_path (Path): Path to model weights
        
    Returns:
        Path: Resolved model path
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    model_path = Path(model_path).resolve()
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"\n❌ RIPE weights not found at:\n{model_path}\n\n"
            "👉 Please download:\n"
            "https://cvg.hhi.fraunhofer.de/RIPE/ripe_weights.pth\n"
            "and place it inside:\n"
            "RIPE-master/weights/ripe_weights.pth\n"
        )
    
    return model_path


def vgg_hyper(
    model_path: Optional[Path] = None,
    desc_shares: Optional[Any] = None,
    device: Optional[torch.device] = None,
    use_cache: bool = True,
    dtype: torch.dtype = torch.float32,
    eval_mode: bool = True,
) -> nn.Module:
    """Initialize RIPE model with VGG backbone and hypercolumn features.
    
    Optimized version with caching, device management, and memory efficiency.
    
    Args:
        model_path (Path): Path to pretrained weights. If None, uses default location.
        desc_shares (Any): Descriptor sharing configuration
        device (torch.device): Device to load model on (auto-detect if None)
        use_cache (bool): Cache model weights for repeated instantiation
        dtype (torch.dtype): Data type for model (float32 or float16)
        eval_mode (bool): Load in evaluation mode (disables dropout, batch norm updates)
        
    Returns:
        nn.Module: Initialized RIPE model
        
    Example:
        >>> model = vgg_hyper(device=torch.device('cuda'))
        >>> kpts, descs, scores = model.detectAndCompute(image_tensor)
    """
    
    # Auto-detect device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine model path
    if model_path is None:
        model_path = (
            Path(__file__).resolve()
            .parent.parent.parent
            / "weights"
            / "ripe_weights.pth"
        )
    
    model_path = _validate_model_path(model_path)
    
    # Check cache
    cache_key = f"{model_path}_{desc_shares}_{dtype}"
    cached_weights = ModelCache.get(cache_key) if use_cache else None
    
    try:
        # Import models here to avoid circular imports
        from ripe.models.backbones.vgg import VGG
        from ripe.models.ripe import RIPE
        from ripe.models.upsampler.hypercolumn_features import HyperColumnFeatures
        
        logger.info(f"✅ Loading RIPE model from: {model_path}")
        
        # Initialize model architecture
        backbone = VGG(pretrained=False)
        upsampler = HyperColumnFeatures()
        
        extractor = RIPE(
            net=backbone,
            upsampler=upsampler,
            desc_shares=desc_shares,
        )
        
        # Load weights
        if cached_weights is not None:
            logger.info("Loading weights from cache")
            extractor.load_state_dict(cached_weights)
        else:
            logger.info(f"Loading weights from disk: {model_path}")
            state_dict = torch.load(model_path, map_location='cpu')
            extractor.load_state_dict(state_dict)
            
            # Cache weights if enabled
            if use_cache:
                ModelCache.set(cache_key, state_dict)
        
        # Move to device
        extractor = extractor.to(device)
        
        # Convert dtype if needed
        if dtype != torch.float32:
            logger.info(f"Converting model to {dtype}")
            extractor = extractor.to(dtype=dtype)
        
        # Set evaluation mode
        if eval_mode:
            extractor.eval()
            # Disable gradients for inference
            for param in extractor.parameters():
                param.requires_grad = False
        
        logger.info(f"Model successfully loaded on {device}")
        return extractor
    
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        raise ImportError(
            "Failed to import RIPE model components. "
            "Ensure ripe package is properly installed."
        ) from e
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise


class VGGHyperFactory:
    """Factory for creating RIPE models with different configurations."""
    
    _models: Dict[str, nn.Module] = {}
    
    @classmethod
    def create(
        cls,
        name: str = "default",
        model_path: Optional[Path] = None,
        **kwargs,
    ) -> nn.Module:
        """Create or retrieve a RIPE model.
        
        Args:
            name (str): Configuration name
            model_path (Path): Path to model weights
            **kwargs: Additional arguments for vgg_hyper
            
        Returns:
            nn.Module: RIPE model instance
        """
        if name not in cls._models:
            logger.info(f"Creating new RIPE model: {name}")
            model = vgg_hyper(model_path=model_path, **kwargs)
            cls._models[name] = model
        else:
            logger.info(f"Retrieving cached RIPE model: {name}")
        
        return cls._models[name]
    
    @classmethod
    def get(cls, name: str = "default") -> Optional[nn.Module]:
        """Get existing model without creating."""
        return cls._models.get(name)
    
    @classmethod
    def register(cls, name: str, model: nn.Module) -> None:
        """Register a custom model."""
        cls._models[name] = model
    
    @classmethod
    def clear(cls, name: Optional[str] = None) -> None:
        """Clear model cache."""
        if name is None:
            cls._models.clear()
        elif name in cls._models:
            del cls._models[name]


def create_ripe_extractor(
    model_path: Optional[Path] = None,
    fp16: bool = False,
    eval_mode: bool = True,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Simplified factory function for creating RIPE model.
    
    Args:
        model_path (Path): Path to weights
        fp16 (bool): Use float16 for faster inference
        eval_mode (bool): Enable evaluation mode
        device (torch.device): Target device
        
    Returns:
        nn.Module: RIPE model
    """
    dtype = torch.float16 if fp16 else torch.float32
    
    return vgg_hyper(
        model_path=model_path,
        device=device,
        dtype=dtype,
        eval_mode=eval_mode,
        use_cache=True,
    )


# Convenience alias
get_ripe_model = create_ripe_extractor