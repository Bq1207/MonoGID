"""
BiFPN模块 - 双向特征金字塔网络

实现要点:
1. BiFPN block: 1x1 lateral conv + 可学习权重融合（EfficientDet风格），重复2次
2. GeometryBranch: 从coarse depth计算几何信息（梯度或法向量）并注入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np


class LearnableWeightFusion(nn.Module):
    """
    可学习权重融合函数（EfficientDet BiFPN风格）
    使用softplus或relu后归一化
    """
    def __init__(self, num_inputs: int = 2, activation: str = 'relu', eps: float = 1e-4):
        super(LearnableWeightFusion, self).__init__()
        self.num_inputs = num_inputs
        self.activation = activation
        self.eps = eps
        
        self.weights = nn.Parameter(torch.ones(num_inputs))
        
        if activation == 'relu':
            self.act = nn.ReLU(inplace=False)
        elif activation == 'softplus':
            self.act = nn.Softplus()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) == self.num_inputs, f"Expected {self.num_inputs} inputs, got {len(inputs)}"
        weights = self.act(self.weights)
        weights = weights / (weights.sum() + self.eps)
        fused = sum(w * x for w, x in zip(weights, inputs))
        return fused


class BiFPNBlock(nn.Module):
    """
    BiFPN单个融合块
    实现top-down和bottom-up双向融合，带可学习权重
    """
    def __init__(self, channels: int, num_levels: int = 4, use_weight_fusion: bool = True):
        super(BiFPNBlock, self).__init__()
        self.channels = channels
        self.num_levels = num_levels
        self.use_weight_fusion = use_weight_fusion
        
        self.topdown_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_levels - 1)
        ])
        
        self.bottomup_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_levels - 1)
        ])
        
        if use_weight_fusion:
            self.topdown_weights = nn.ModuleList([
                LearnableWeightFusion(num_inputs=2) for _ in range(num_levels - 1)
            ])
            self.bottomup_weights = nn.ModuleList([
                LearnableWeightFusion(num_inputs=2) for _ in range(num_levels - 1)
            ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Top-down路径
        td_features = [features[-1]]
        for i in range(self.num_levels - 2, -1, -1):
            upsampled = F.interpolate(
                td_features[-1],
                size=features[i].shape[2:],
                mode='bilinear',
                align_corners=True
            )
            if self.use_weight_fusion:
                fused = self.topdown_weights[i]([upsampled, features[i]])
            else:
                fused = upsampled + features[i]
            fused = self.topdown_convs[i](fused)
            td_features.append(fused)
        
        td_features = td_features[::-1]
        
        # Bottom-up路径
        bu_features = [td_features[0]]
        for i in range(1, self.num_levels):
            downsampled = F.interpolate(
                bu_features[-1],
                size=td_features[i].shape[2:],
                mode='bilinear',
                align_corners=True
            )
            if self.use_weight_fusion:
                fused = self.bottomup_weights[i-1]([downsampled, td_features[i]])
            else:
                fused = downsampled + td_features[i]
            fused = self.bottomup_convs[i-1](fused)
            bu_features.append(fused)
        
        return bu_features


class GeometryBranch(nn.Module):
    """
    几何信息分支：从 coarse depth 计算几何信息（梯度或法向量）并编码

    流程:
    1. 从最深层BiFPN特征生成 coarse depth
    2. 计算几何信息（梯度或法向量）
    3. Conv3x3编码几何信息到BiFPN通道数
    4. 在每次BiFPN fusion后注入
    """
    def __init__(
        self,
        channels: int,
        geo_type: str = "gradient",
        use_gated_fusion: bool = True
    ):
        super(GeometryBranch, self).__init__()
        self.channels = channels
        self.geo_type = geo_type
        self.use_gated_fusion = use_gated_fusion
        
        self.depth_decode = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1)
        )
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        geo_in_channels = 1 if geo_type == "gradient" else 3
        
        self.geo_encoder = nn.Sequential(
            nn.Conv2d(geo_in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        if use_gated_fusion:
            self.gate_conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
    
    def compute_gradient(self, depth: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return grad_magnitude
    
    def depth_to_normal(self, depth: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        grad_norm = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1.0)
        Nx = -grad_x / grad_norm
        Ny = -grad_y / grad_norm
        Nz = 1.0 / grad_norm
        return torch.cat([Nx, Ny, Nz], dim=1)
    
    def forward(self, deepest_feat: torch.Tensor, target_sizes: List[Tuple[int, int]]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        d_coarse = self.depth_decode(deepest_feat)
        
        if self.geo_type == "gradient":
            geo_info = self.compute_gradient(d_coarse)
        else:
            geo_info = self.depth_to_normal(d_coarse)
        
        geo_feat_base = self.geo_encoder(geo_info)
        
        geo_features = []
        for h, w in target_sizes:
            if geo_feat_base.shape[2:] != (h, w):
                geo_feat = F.interpolate(
                    geo_feat_base, size=(h, w),
                    mode='bilinear', align_corners=True
                )
            else:
                geo_feat = geo_feat_base
            geo_features.append(geo_feat)
        
        return geo_features, d_coarse
    
    def fuse_geometry(self, features: List[torch.Tensor], geo_features: List[torch.Tensor]) -> List[torch.Tensor]:
        fused = []
        for feat, geo_feat in zip(features, geo_features):
            if self.use_gated_fusion:
                gate = self.gate_conv(geo_feat)
                fused_feat = feat * gate + geo_feat * (1 - gate)
            else:
                fused_feat = feat + geo_feat
            fused.append(fused_feat)
        return fused


class EnhancedBiFPN(nn.Module):
    """
    增强BiFPN完整实现

    包含:
    - BiFPN blocks (重复num_bifpn_blocks次)
    - GeometryBranch (可选)：在每次BiFPN fusion后注入几何信息
    """
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_bifpn_blocks: int = 2,
        use_geo_injection: bool = False,
        geo_type: str = "gradient",
        geo_fusion_type: str = "gated"
    ):
        super(EnhancedBiFPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = len(in_channels)
        self.num_bifpn_blocks = num_bifpn_blocks
        self.use_geo_injection = use_geo_injection
        
        # 1x1 lateral conv统一通道
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
            for in_ch in in_channels
        ])
        
        # BiFPN blocks (重复num_bifpn_blocks次)
        self.bifpn_blocks = nn.ModuleList([
            BiFPNBlock(out_channels, num_levels=self.num_levels, use_weight_fusion=True)
            for _ in range(num_bifpn_blocks)
        ])
        
        # GeometryBranch (在BiFPN fusion后注入)
        if use_geo_injection:
            self.geo_branch = GeometryBranch(
                channels=out_channels,
                geo_type=geo_type,
                use_gated_fusion=(geo_fusion_type == "gated")
            )
    
    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        # Lateral conv统一通道
        bifpn_features = [
            conv(feat) for conv, feat in zip(self.lateral_convs, features)
        ]
        
        d_coarse = None
        
        for bifpn_block in self.bifpn_blocks:
            bifpn_features = bifpn_block(bifpn_features)
            
            if self.use_geo_injection:
                target_sizes = [feat.shape[2:] for feat in bifpn_features]
                geo_features, d_coarse = self.geo_branch(bifpn_features[-1], target_sizes)
                bifpn_features = self.geo_branch.fuse_geometry(bifpn_features, geo_features)
        
        return bifpn_features, d_coarse


class ChannelPreservingEnhancedBiFPN(nn.Module):
    """
    保持通道数的增强BiFPN包装器
    与现有FPN接口兼容
    """
    def __init__(
        self,
        in_out_channels: List[int],
        unified_channels: int = 256,
        num_bifpn_blocks: int = 2,
        use_geo_injection: bool = False,
        geo_type: str = "gradient",
        geo_fusion_type: str = "gated",
        **kwargs
    ):
        super(ChannelPreservingEnhancedBiFPN, self).__init__()
        self.in_out_channels = in_out_channels
        self.unified_channels = unified_channels
        self.use_geo_injection = use_geo_injection
        
        self.to_unified = nn.ModuleList([
            nn.Conv2d(ch, unified_channels, kernel_size=1, bias=False)
            for ch in in_out_channels
        ])
        
        self.bifpn = EnhancedBiFPN(
            [unified_channels] * len(in_out_channels),
            out_channels=unified_channels,
            num_bifpn_blocks=num_bifpn_blocks,
            use_geo_injection=use_geo_injection,
            geo_type=geo_type,
            geo_fusion_type=geo_fusion_type
        )
        
        self.from_unified = nn.ModuleList([
            nn.Conv2d(unified_channels, ch, kernel_size=1, bias=False)
            for ch in in_out_channels
        ])
    
    def forward(self, features: List[torch.Tensor], input_image: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        unified = [conv(feat) for conv, feat in zip(self.to_unified, features)]
        bifpn_out, d_coarse = self.bifpn(unified)
        outputs = [conv(feat) for conv, feat in zip(self.from_unified, bifpn_out)]
        return outputs, d_coarse
