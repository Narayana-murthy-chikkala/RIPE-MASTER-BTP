# Optimized VGG utilities with memory efficiency improvements
# Adapted from DeDoDe repository with performance enhancements

import torch
import torch.nn as nn
import torchvision.models as tvm
from typing import List, Tuple, Optional
from ripe import utils

log = utils.get_pylogger(__name__)


class ConvRefiner(nn.Module):
    """Optimized ConvRefiner with memory-efficient operations.
    
    Args:
        in_dim (int): Input channel dimension
        hidden_dim (int): Hidden channel dimension
        out_dim (int): Output channel dimension
        dw (bool): Use depthwise convolutions for memory efficiency
        kernel_size (int): Kernel size for hidden blocks
        hidden_blocks (int): Number of hidden blocks
        residual (bool): Use residual connections
    """
    
    def __init__(
        self,
        in_dim: int = 6,
        hidden_dim: int = 16,
        out_dim: int = 2,
        dw: bool = True,
        kernel_size: int = 5,
        hidden_blocks: int = 5,
        residual: bool = False,
    ):
        super().__init__()
        self.residual = residual
        self.hidden_dim = hidden_dim
        
        # Initial 1×1 convolution for dimensionality adjustment
        self.block1 = self._create_block(
            in_dim,
            hidden_dim,
            dw=False,
            kernel_size=1,
        )
        
        # Stack of hidden blocks (depthwise or standard conv)
        self.hidden_blocks = nn.Sequential(
            *[
                self._create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                )
                for _ in range(hidden_blocks)
            ]
        )
        
        # Final output convolution
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=True)
        
        # Initialize weights for better convergence
        self._init_weights()

    def _create_block(
        self,
        in_dim: int,
        out_dim: int,
        dw: bool = True,
        kernel_size: int = 5,
        bias: bool = True,
        norm_type: nn.Module = nn.BatchNorm2d,
    ) -> nn.Sequential:
        """Create a conv block with optional depthwise convolution."""
        num_groups = in_dim if dw else 1
        
        if dw:
            assert out_dim % in_dim == 0, "out_dim must be divisible by in_dim for depthwise"
        
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        
        norm = (
            norm_type(out_dim) 
            if norm_type is nn.BatchNorm2d 
            else norm_type(num_channels=out_dim)
        )
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=bias)
        
        return nn.Sequential(conv1, norm, relu, conv2)

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection.
        
        Args:
            feats (torch.Tensor): Input features (B, C, H, W)
            
        Returns:
            torch.Tensor: Output features (B, out_dim, H, W)
        """
        x0 = self.block1(feats)
        x = self.hidden_blocks(x0)
        
        if self.residual:
            # Normalize by dividing by sqrt(2) to maintain activation scale
            x = (x + x0) / 1.4
        
        x = self.out_conv(x)
        return x


class Decoder(nn.Module):
    """Progressive decoder with feature refinement.
    
    Args:
        layers (nn.ModuleDict): Dictionary of ConvRefiners indexed by scale
        super_resolution (bool): Enable super-resolution mode
        num_prototypes (int): Number of output prototypes (detection channels)
    """
    
    def __init__(
        self,
        layers: nn.ModuleDict,
        *args,
        super_resolution: bool = False,
        num_prototypes: int = 1,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.scales = list(layers.keys())
        self.super_resolution = super_resolution
        self.num_prototypes = num_prototypes

    def forward(
        self,
        features: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        scale: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode features at a specific scale.
        
        Args:
            features (torch.Tensor): Input features (B, C, H, W)
            context (torch.Tensor): Context from previous scale
            scale (str): Scale identifier ("8", "4", "2", "1")
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (logits, context)
        """
        # Concatenate features and context if available
        if context is not None:
            features = torch.cat((features, context), dim=1)
        
        # Apply ConvRefiner at this scale
        stuff = self.layers[scale](features)
        
        # Split output: first channels are logits, rest is context
        logits = stuff[:, : self.num_prototypes]
        context_out = stuff[:, self.num_prototypes :]
        
        return logits, context_out


class VGG19(nn.Module):
    """Optimized VGG19 encoder for feature extraction.
    
    Uses only first 40 layers (up to the last maxpool) for efficiency.
    Supports custom input channels for multi-spectral data.
    
    Args:
        pretrained (bool): Load pretrained ImageNet weights
        num_input_channels (int): Number of input channels (default: 3 for RGB)
    """
    
    def __init__(self, pretrained: bool = False, num_input_channels: int = 3) -> None:
        super().__init__()
        
        # Load pretrained VGG19 with batch normalization (first 40 layers)
        full_model = tvm.vgg19_bn(pretrained=pretrained)
        self.layers = nn.ModuleList(full_model.features[:40])
        
        # Adapt first layer if different input channels
        if num_input_channels != 3:
            log.info(f"Converting input channels from 3 to {num_input_channels}")
            self.layers[0] = nn.Conv2d(num_input_channels, 64, kernel_size=3, padding=1)
            if pretrained:
                # Average the pretrained weights across color channels
                with torch.no_grad():
                    original_weight = full_model.features[0].weight
                    self.layers[0].weight.data = original_weight.mean(dim=1, keepdim=True)

    def get_dim_layers(self) -> List[int]:
        """Get channel dimensions at each scale."""
        return [64, 128, 256, 512]

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """Extract multi-scale features.
        
        Args:
            x (torch.Tensor): Input image (B, C, H, W)
            
        Returns:
            Tuple: (features_list, sizes_list)
                - features_list: Features at 1/1, 1/2, 1/4, 1/8 scales
                - sizes_list: Spatial dimensions for each feature map
        """
        feats = []
        sizes = []
        
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                # Save feature map before pooling
                feats.append(x)
                sizes.append(x.shape[-2:])
            x = layer(x)
        
        return feats, sizes


class VGG(nn.Module):
    """Generic VGG encoder supporting different model sizes.
    
    Args:
        size (str): Model size ("11", "13", or "19")
        pretrained (bool): Load pretrained ImageNet weights
    """
    
    def __init__(self, size: str = "19", pretrained: bool = False) -> None:
        super().__init__()
        
        # Select appropriate model based on size
        if size == "11":
            full_model = tvm.vgg11_bn(pretrained=pretrained)
            self.layers = nn.ModuleList(full_model.features[:22])
        elif size == "13":
            full_model = tvm.vgg13_bn(pretrained=pretrained)
            self.layers = nn.ModuleList(full_model.features[:28])
        elif size == "19":
            full_model = tvm.vgg19_bn(pretrained=pretrained)
            self.layers = nn.ModuleList(full_model.features[:40])
        else:
            raise ValueError(f"VGG size must be 11, 13, or 19, got {size}")

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """Extract multi-scale features.
        
        Args:
            x (torch.Tensor): Input image (B, C, H, W)
            
        Returns:
            Tuple: (features_list, sizes_list)
        """
        feats = []
        sizes = []
        
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                feats.append(x)
                sizes.append(x.shape[-2:])
            x = layer(x)
        
        return feats, sizes


# Memory-efficient inference mode
class VGG19Inference(VGG19):
    """VGG19 optimized for inference with gradient checkpointing disabled."""
    
    def __init__(self, pretrained: bool = True, num_input_channels: int = 3):
        super().__init__(pretrained=pretrained, num_input_channels=num_input_channels)
        # Disable gradients for all parameters (inference only)
        for param in self.parameters():
            param.requires_grad = False