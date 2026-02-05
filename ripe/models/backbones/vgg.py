# Optimized VGG Backbone Implementation
# Supports multi-scale feature extraction with flexible modes

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
from .backbone_base import BackboneBase
from .vgg_utils import VGG19, ConvRefiner, Decoder


class VGG(BackboneBase):
    """Optimized VGG-based backbone for keypoint detection and description.
    
    Supports three modes:
    - "dect": Detection only (1 output channel for heatmap)
    - "desc": Description only (256 output channels for descriptors)
    - "dect+desc": Combined detection and description (257 channels)
    
    Architecture:
    - Encoder: VGG19 with batch normalization (40 layers)
    - Decoder: Progressive upsampling with ConvRefiners at 4 scales
    - Multi-scale feature fusion from coarse to fine
    
    Args:
        nchannels (int): Number of input channels (default: 3 for RGB)
        pretrained (bool): Load ImageNet pretrained weights
        use_instance_norm (bool): Use instance normalization
        mode (str): Output mode - "dect", "desc", or "dect+desc"
    """
    
    def __init__(
        self,
        nchannels: int = 3,
        pretrained: bool = True,
        use_instance_norm: bool = True,
        mode: str = "dect",
    ):
        super().__init__(nchannels=nchannels, use_instance_norm=use_instance_norm)
        
        self.nchannels = nchannels
        self.mode = mode
        
        # Validate mode
        if self.mode not in ["dect", "desc", "dect+desc"]:
            raise ValueError(f"mode must be 'dect', 'desc', or 'dect+desc', got '{mode}'")
        
        # Get architecture parameters based on mode
        num_output_channels, hidden_blocks = self._get_mode_params(mode)
        
        # Create ConvRefiners for each scale
        conv_refiner = self._create_conv_refiner(num_output_channels, hidden_blocks)
        
        # Initialize encoder and decoder
        self.encoder = VGG19(pretrained=pretrained, num_input_channels=nchannels)
        self.decoder = Decoder(conv_refiner, num_prototypes=num_output_channels)
        
        # Cache for efficiency
        self._decoder_scales = list(conv_refiner.keys())

    @staticmethod
    def _get_mode_params(mode: str) -> Tuple[int, int]:
        """Get output channels and hidden blocks based on mode.
        
        Args:
            mode (str): Operating mode
            
        Returns:
            Tuple[int, int]: (num_output_channels, num_hidden_blocks)
        """
        if mode == "dect":
            return 1, 8
        elif mode == "desc":
            return 256, 5
        elif mode == "dect+desc":
            return 257, 8
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def _create_conv_refiner(
        num_output_channels: int,
        hidden_blocks: int,
    ) -> nn.ModuleDict:
        """Create ConvRefiners for all scales.
        
        Args:
            num_output_channels (int): Output channels for ConvRefiners
            hidden_blocks (int): Number of hidden blocks per refiner
            
        Returns:
            nn.ModuleDict: Dictionary of ConvRefiners indexed by scale
        """
        return nn.ModuleDict(
            {
                "8": ConvRefiner(
                    in_dim=512,
                    hidden_dim=512,
                    out_dim=256 + num_output_channels,
                    hidden_blocks=hidden_blocks,
                    residual=True,
                ),
                "4": ConvRefiner(
                    in_dim=256 + 256,
                    hidden_dim=256,
                    out_dim=128 + num_output_channels,
                    hidden_blocks=hidden_blocks,
                    residual=True,
                ),
                "2": ConvRefiner(
                    in_dim=128 + 128,
                    hidden_dim=128,
                    out_dim=64 + num_output_channels,
                    hidden_blocks=hidden_blocks,
                    residual=True,
                ),
                "1": ConvRefiner(
                    in_dim=64 + 64,
                    hidden_dim=64,
                    out_dim=1 + num_output_channels,
                    hidden_blocks=hidden_blocks,
                    residual=True,
                ),
            }
        )

    def get_dim_layers_encoder(self) -> List[int]:
        """Get channel dimensions from encoder at each scale."""
        return self.encoder.get_dim_layers()

    def _forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through encoder and decoder.
        
        Args:
            x (torch.Tensor): Input image (B, C, H, W)
            
        Returns:
            Dict[str, torch.Tensor]: Output dictionary with mode-specific keys:
                - "dect": {"heatmap", "coarse_descs"}
                - "desc": {"fine_descs", "coarse_descs"}
                - "dect+desc": {"heatmap", "fine_descs", "coarse_descs"}
        """
        # Encoder: Extract multi-scale features
        features, sizes = self.encoder(x)
        
        # Decoder: Progressive refinement from coarse to fine
        output = 0
        context = None
        scales = self.decoder.scales
        
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            # Apply decoder at this scale
            delta_descriptor, context = self.decoder(
                feature_map,
                scale=scale,
                context=context,
            )
            
            # Accumulate outputs
            output = output + delta_descriptor
            
            # Interpolate for next scale (if not the last)
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                
                # Efficient bilinear interpolation
                output = F.interpolate(
                    output,
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
                context = F.interpolate(
                    context,
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
        
        # Package output based on mode
        if self.mode == "dect":
            return {
                "heatmap": output,
                "coarse_descs": features,
            }
        elif self.mode == "desc":
            return {
                "fine_descs": output,
                "coarse_descs": features,
            }
        elif self.mode == "dect+desc":
            # Split output: first channel is heatmap, rest are descriptors
            heatmap = output[:, :1].contiguous()
            descriptors = output[:, 1:].contiguous()
            
            return {
                "heatmap": heatmap,
                "fine_descs": descriptors,
                "coarse_descs": features,
            }
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor] | Tuple:
        """Public forward method with optional dict return.
        
        Args:
            x (torch.Tensor): Input image (B, C, H, W)
            return_dict (bool): Return dictionary or tuple
            
        Returns:
            Dict or Tuple: Model outputs
        """
        output = self._forward(x)
        
        if not return_dict:
            if self.mode == "dect":
                return output["heatmap"], output["coarse_descs"]
            elif self.mode == "desc":
                return output["fine_descs"], output["coarse_descs"]
            else:
                return (
                    output["heatmap"],
                    output["fine_descs"],
                    output["coarse_descs"],
                )
        
        return output


class VGGLite(VGG):
    """Lightweight VGG variant using VGG13 for faster inference.
    
    Trades accuracy for speed - suitable for real-time applications.
    """
    
    def __init__(
        self,
        nchannels: int = 3,
        pretrained: bool = True,
        use_instance_norm: bool = True,
        mode: str = "dect",
    ):
        # Call BackboneBase.__init__ directly to avoid VGG.__init__
        nn.Module.__init__(self)
        
        self.nchannels = nchannels
        self.mode = mode
        
        if self.mode not in ["dect", "desc", "dect+desc"]:
            raise ValueError(f"mode must be 'dect', 'desc', or 'dect+desc', got '{mode}'")
        
        num_output_channels, hidden_blocks = self._get_mode_params(mode)
        conv_refiner = self._create_conv_refiner(num_output_channels, hidden_blocks)
        
        # Use lightweight encoder
        from .vgg_utils import VGG as GenericVGG
        self.encoder = GenericVGG(size="13", pretrained=pretrained)
        self.decoder = Decoder(conv_refiner, num_prototypes=num_output_channels)
        
        self._decoder_scales = list(conv_refiner.keys())
    
    @staticmethod
    def _get_mode_params(mode: str) -> Tuple[int, int]:
        return VGG._get_mode_params(mode)
    
    @staticmethod
    def _create_conv_refiner(num_output_channels: int, hidden_blocks: int):
        return VGG._create_conv_refiner(num_output_channels, hidden_blocks)


# Mixed precision support
class VGGMixedPrecision(VGG):
    """VGG with automatic mixed precision for faster inference."""
    
    def forward(self, x: torch.Tensor, return_dict: bool = True):
        """Forward with automatic mixed precision."""
        with torch.cuda.amp.autocast(enabled=x.is_cuda):
            output = self._forward(x)
        
        if not return_dict:
            if self.mode == "dect":
                return output["heatmap"], output["coarse_descs"]
            elif self.mode == "desc":
                return output["fine_descs"], output["coarse_descs"]
            else:
                return (
                    output["heatmap"],
                    output["fine_descs"],
                    output["coarse_descs"],
                )
        
        return output