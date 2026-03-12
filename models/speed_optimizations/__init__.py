"""
Speed Optimization Modules for EndoDAC-FPN
推理加速优化模块

包含四个核心创新点:
1. LoRA权重融合 (LoRA Weight Fusion) - 消除推理时额外计算
2. 自适应Token剪枝 (Adaptive Token Pruning) - 减少注意力计算量
3. 轻量级动态FPN (Lightweight Dynamic FPN) - 高效多尺度融合
4. 高效ISA模块 (Efficient ISA Module) - 参数共享注意力

使用方法:
    # 快速优化现有模型
    from models.speed_optimizations import optimize_for_inference
    fast_model = optimize_for_inference(model)

    # 使用FastEndoDAC
    from models.speed_optimizations import FastEndoDAC
    fast_model = FastEndoDAC.from_pretrained(original_model)
"""

from .lora_fusion import (
    LoRAFusionMixin,
    fuse_lora_weights,
    unfuse_lora_weights,
    DVLinearFusable,
    convert_to_fusable_lora
)
from .token_pruning import (
    AdaptiveTokenPruning,
    TokenPruningViT,
    TokenImportanceEstimator,
    TokenPruningAttention
)
from .efficient_fpn import (
    EfficientDynamicFPN,
    DepthwiseFPNBlock,
    LazyFusionFPN,
    ChannelPreservingEfficientFPN,
    DepthwiseSeparableConv,
    DynamicRoutingModule
)
from .efficient_isa import (
    EfficientISA,
    SharedISA,
    MultiScaleISA,
    ConditionalISA,
    LinearAttention,
    EfficientSEAttention
)
from .fast_endodac import (
    FastEndoDAC,
    FastDPTHead,
    FastViTEncoder,
    optimize_for_inference
)

__all__ = [
    # LoRA Fusion
    'LoRAFusionMixin', 'fuse_lora_weights', 'unfuse_lora_weights', 'DVLinearFusable', 'convert_to_fusable_lora',
    # Token Pruning
    'AdaptiveTokenPruning', 'TokenPruningViT', 'TokenImportanceEstimator', 'TokenPruningAttention',
    # Efficient FPN
    'EfficientDynamicFPN', 'DepthwiseFPNBlock', 'LazyFusionFPN',
    'ChannelPreservingEfficientFPN', 'DepthwiseSeparableConv', 'DynamicRoutingModule',
    # Efficient ISA
    'EfficientISA', 'SharedISA', 'MultiScaleISA', 'ConditionalISA',
    'LinearAttention', 'EfficientSEAttention',
    # Fast Model
    'FastEndoDAC', 'FastDPTHead', 'FastViTEncoder', 'optimize_for_inference'
]
