"""
创新点3: 轻量级动态FPN (Lightweight Dynamic FPN)

核心思想:
- 传统FPN使用标准卷积，计算量大
- 提出使用深度可分离卷积 + 动态路由的高效FPN
- 根据输入特征动态调整融合权重，减少冗余计算

技术贡献:
1. 深度可分离卷积替代标准卷积，参数量减少8-9倍
2. 动态路由机制：根据特征内容自适应选择融合路径
3. 特征重用策略：跨层共享部分卷积核
4. 延迟融合：将多次小融合合并为一次大融合

预期加速: 15-20% (FPN模块加速50%以上)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积

    将标准卷积分解为:
    1. 深度卷积 (Depthwise): 每个通道独立卷积
    2. 逐点卷积 (Pointwise): 1x1卷积混合通道

    参数量: C_in * K * K + C_in * C_out (vs 标准卷积: C_in * C_out * K * K)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        use_bn: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True)
    ):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=bias
        )

        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DynamicRoutingModule(nn.Module):
    """
    动态路由模块

    根据输入特征动态生成融合权重，实现自适应特征融合
    """
    def __init__(self, channels: int, num_inputs: int = 2, reduction: int = 16):
        super().__init__()
        self.num_inputs = num_inputs

        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 路由权重生成器
        self.router = nn.Sequential(
            nn.Linear(channels * num_inputs, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, num_inputs),
            nn.Softmax(dim=-1)
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        动态融合多个特征

        Args:
            features: 特征列表，每个形状为 [B, C, H, W]

        Returns:
            fused: 融合后的特征 [B, C, H, W]
        """
        assert len(features) == self.num_inputs

        B = features[0].shape[0]

        # 提取全局特征
        global_feats = []
        for feat in features:
            global_feats.append(self.global_pool(feat).view(B, -1))

        # 拼接全局特征
        concat_feats = torch.cat(global_feats, dim=-1)  # [B, C*num_inputs]

        # 生成路由权重
        weights = self.router(concat_feats)  # [B, num_inputs]

        # 加权融合
        fused = torch.zeros_like(features[0])
        for i, feat in enumerate(features):
            fused = fused + feat * weights[:, i:i+1, None, None]

        return fused


class DepthwiseFPNBlock(nn.Module):
    """
    深度可分离FPN融合块

    使用深度可分离卷积进行特征融合，大幅减少计算量
    """
    def __init__(self, channels: int, use_dynamic_routing: bool = True):
        super().__init__()
        self.use_dynamic_routing = use_dynamic_routing

        # 深度可分离卷积进行特征细化
        self.refine = DepthwiseSeparableConv(
            channels, channels,
            kernel_size=3, padding=1
        )

        # 动态路由 (可选)
        if use_dynamic_routing:
            self.router = DynamicRoutingModule(channels, num_inputs=2)

    def forward(
        self,
        high_level: torch.Tensor,
        low_level: torch.Tensor
    ) -> torch.Tensor:
        """
        融合高层和低层特征

        Args:
            high_level: 高层特征 (小分辨率) [B, C, H1, W1]
            low_level: 低层特征 (大分辨率) [B, C, H2, W2]

        Returns:
            fused: 融合后的特征 [B, C, H2, W2]
        """
        # 上采样高层特征
        upsampled = F.interpolate(
            high_level,
            size=low_level.shape[2:],
            mode='bilinear',
            align_corners=True
        )

        # 融合
        if self.use_dynamic_routing:
            fused = self.router([upsampled, low_level])
        else:
            fused = upsampled + low_level

        # 细化
        fused = self.refine(fused)

        return fused


class EfficientDynamicFPN(nn.Module):
    """
    高效动态FPN

    特点:
    1. 深度可分离卷积: 减少参数量和计算量
    2. 动态路由: 自适应特征融合
    3. 共享卷积核: 跨层参数共享
    4. 延迟融合: 减少中间计算
    """
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        use_dynamic_routing: bool = True,
        share_weights: bool = True,
        use_bottom_up: bool = False
    ):
        """
        Args:
            in_channels: 输入通道数列表 [C1, C2, C3, C4]
            out_channels: 统一输出通道数
            use_dynamic_routing: 是否使用动态路由
            share_weights: 是否共享融合块权重
            use_bottom_up: 是否使用自底向上路径
        """
        super().__init__()
        self.num_levels = len(in_channels)
        self.out_channels = out_channels
        self.use_bottom_up = use_bottom_up

        # 横向连接: 1x1卷积对齐通道
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
            for in_ch in in_channels
        ])

        # 自顶向下融合块
        if share_weights:
            # 共享权重: 所有层使用同一个融合块
            self.shared_fpn_block = DepthwiseFPNBlock(out_channels, use_dynamic_routing)
            self.fpn_blocks = None
        else:
            # 独立权重
            self.fpn_blocks = nn.ModuleList([
                DepthwiseFPNBlock(out_channels, use_dynamic_routing)
                for _ in range(self.num_levels - 1)
            ])
            self.shared_fpn_block = None

        # 自底向上路径 (可选)
        if use_bottom_up:
            self.bottom_up_convs = nn.ModuleList([
                DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                for _ in range(self.num_levels - 1)
            ])

        # 输出细化
        self.output_convs = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(self.num_levels)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        前向传播

        Args:
            features: 多尺度特征列表 [C1, C2, C3, C4]

        Returns:
            fpn_features: FPN输出特征列表
        """
        assert len(features) == self.num_levels

        # Step 1: 横向连接
        lateral_features = [
            conv(feat) for conv, feat in zip(self.lateral_convs, features)
        ]

        # Step 2: 自顶向下路径
        fpn_features = [lateral_features[-1]]  # 最高层直接使用

        for i in range(self.num_levels - 2, -1, -1):
            if self.shared_fpn_block is not None:
                fused = self.shared_fpn_block(fpn_features[-1], lateral_features[i])
            else:
                fused = self.fpn_blocks[i](fpn_features[-1], lateral_features[i])
            fpn_features.append(fused)

        fpn_features = fpn_features[::-1]  # 反转为从低到高

        # Step 3: 自底向上路径 (可选)
        if self.use_bottom_up:
            enhanced = [fpn_features[0]]
            for i in range(1, self.num_levels):
                downsampled = self.bottom_up_convs[i-1](enhanced[-1])
                enhanced.append(downsampled + fpn_features[i])
            fpn_features = enhanced

        # Step 4: 输出细化
        outputs = [
            conv(feat) for conv, feat in zip(self.output_convs, fpn_features)
        ]

        return outputs


class LazyFusionFPN(nn.Module):
    """
    延迟融合FPN

    核心思想: 将多次小规模融合合并为一次大规模融合
    传统FPN: C4 -> C3 -> C2 -> C1 (3次融合)
    延迟融合: 收集所有特征 -> 一次性融合

    优势:
    1. 减少中间计算和内存访问
    2. 更好的特征交互
    3. 适合并行计算
    """
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        fusion_method: str = 'attention'
    ):
        """
        Args:
            in_channels: 输入通道数列表
            out_channels: 输出通道数
            fusion_method: 融合方法 ('concat', 'attention', 'weighted')
        """
        super().__init__()
        self.num_levels = len(in_channels)
        self.out_channels = out_channels
        self.fusion_method = fusion_method

        # 通道对齐
        self.align_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
            for in_ch in in_channels
        ])

        # 融合模块
        if fusion_method == 'concat':
            self.fusion = nn.Sequential(
                nn.Conv2d(out_channels * self.num_levels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif fusion_method == 'attention':
            self.fusion = CrossScaleAttention(out_channels, self.num_levels)
        elif fusion_method == 'weighted':
            self.fusion_weights = nn.Parameter(torch.ones(self.num_levels) / self.num_levels)

        # 输出投影
        self.output_projs = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(self.num_levels)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        前向传播

        Args:
            features: 多尺度特征列表

        Returns:
            outputs: 融合后的多尺度特征
        """
        # Step 1: 通道对齐
        aligned = [conv(feat) for conv, feat in zip(self.align_convs, features)]

        # Step 2: 统一到最大分辨率
        target_size = aligned[0].shape[2:]
        resized = []
        for feat in aligned:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            resized.append(feat)

        # Step 3: 融合
        if self.fusion_method == 'concat':
            concat = torch.cat(resized, dim=1)
            fused = self.fusion(concat)
        elif self.fusion_method == 'attention':
            fused = self.fusion(resized)
        elif self.fusion_method == 'weighted':
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = sum(w * feat for w, feat in zip(weights, resized))

        # Step 4: 生成多尺度输出
        outputs = []
        for i, (proj, original) in enumerate(zip(self.output_projs, features)):
            # 调整到原始分辨率
            if fused.shape[2:] != original.shape[2:]:
                out = F.interpolate(fused, size=original.shape[2:], mode='bilinear', align_corners=True)
            else:
                out = fused
            outputs.append(proj(out))

        return outputs


class CrossScaleAttention(nn.Module):
    """
    跨尺度注意力模块

    用于延迟融合FPN中的特征融合
    """
    def __init__(self, channels: int, num_scales: int, reduction: int = 16):
        super().__init__()
        self.num_scales = num_scales

        # 尺度注意力
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * num_scales, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, num_scales),
            nn.Softmax(dim=-1)
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels * num_scales, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        跨尺度注意力融合

        Args:
            features: 对齐后的特征列表 (相同分辨率)

        Returns:
            fused: 融合后的特征
        """
        B, C, H, W = features[0].shape

        # 拼接特征
        concat = torch.cat(features, dim=1)  # [B, C*num_scales, H, W]

        # 尺度注意力
        scale_weights = self.scale_attention(concat)  # [B, num_scales]

        # 加权融合
        fused = torch.zeros_like(features[0])
        for i, feat in enumerate(features):
            fused = fused + feat * scale_weights[:, i:i+1, None, None]

        # 空间注意力
        spatial_weights = self.spatial_attention(concat)  # [B, 1, H, W]
        fused = fused * spatial_weights

        return fused


class ChannelPreservingEfficientFPN(nn.Module):
    """
    保持通道数的高效FPN包装器

    与原始FPN接口兼容，但内部使用高效实现
    """
    def __init__(
        self,
        in_out_channels: List[int],
        fpn_type: str = 'efficient',
        use_dynamic_routing: bool = True
    ):
        super().__init__()
        self.in_out_channels = in_out_channels
        self.unified_channels = 256

        # 输入转换
        self.to_unified = nn.ModuleList([
            nn.Conv2d(ch, self.unified_channels, kernel_size=1, bias=False)
            for ch in in_out_channels
        ])

        # 高效FPN
        if fpn_type == 'efficient':
            self.fpn = EfficientDynamicFPN(
                [self.unified_channels] * len(in_out_channels),
                self.unified_channels,
                use_dynamic_routing=use_dynamic_routing
            )
        elif fpn_type == 'lazy':
            self.fpn = LazyFusionFPN(
                [self.unified_channels] * len(in_out_channels),
                self.unified_channels,
                fusion_method='attention'
            )
        else:
            raise ValueError(f"Unknown FPN type: {fpn_type}")

        # 输出转换
        self.from_unified = nn.ModuleList([
            nn.Conv2d(self.unified_channels, ch, kernel_size=1, bias=False)
            for ch in in_out_channels
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # 转换到统一通道
        unified = [conv(feat) for conv, feat in zip(self.to_unified, features)]

        # FPN处理
        fpn_out = self.fpn(unified)

        # 转换回原始通道
        outputs = [conv(feat) for conv, feat in zip(self.from_unified, fpn_out)]

        return outputs
