"""
FastEndoDAC: 高速推理版本的EndoDAC模型

整合所有速度优化创新点:
1. LoRA权重融合 - 消除推理时的额外矩阵乘法
2. 自适应Token剪枝 - 减少注意力计算量
3. 轻量级动态FPN - 高效多尺度特征融合
4. 高效ISA模块 - 参数共享的注意力机制

使用方法:
    # 方式1: 从原始模型转换
    from models.speed_optimizations import FastEndoDAC
    fast_model = FastEndoDAC.from_pretrained(original_model)

    # 方式2: 直接创建
    fast_model = FastEndoDAC(
        backbone_size="base",
        use_token_pruning=True,
        use_efficient_fpn=True,
        use_efficient_isa=True
    )
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import time

from .lora_fusion import fuse_lora_weights, DVLinearFusable
from .token_pruning import AdaptiveTokenPruning, TokenImportanceEstimator
from .efficient_fpn import EfficientDynamicFPN, ChannelPreservingEfficientFPN
from .efficient_isa import EfficientISA, SharedISA, MultiScaleISA


class FastDPTHead(nn.Module):
    """
    高速DPT解码头

    优化点:
    1. 使用高效FPN替代原始FPN
    2. 使用共享ISA替代独立ISA
    3. 融合批归一化层
    """
    def __init__(
        self,
        in_channels: int,
        features: int = 128,
        out_channels: List[int] = [96, 192, 384, 768],
        use_efficient_fpn: bool = True,
        use_efficient_isa: bool = True,
        fpn_type: str = 'efficient'
    ):
        super().__init__()

        self.out_channels = out_channels

        # 投影层
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_ch, kernel_size=1, stride=1, padding=0)
            for out_ch in out_channels
        ])

        # 调整层
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)
        ])

        # 高效FPN
        if use_efficient_fpn:
            self.fpn = ChannelPreservingEfficientFPN(out_channels, fpn_type=fpn_type)
        else:
            self.fpn = None

        # Conv Neck
        self.layer_rn = nn.ModuleList([
            nn.Conv2d(out_ch, features, kernel_size=3, stride=1, padding=1, bias=False)
            for out_ch in out_channels
        ])

        # 高效ISA (共享参数)
        if use_efficient_isa:
            self.isa = SharedISA(features, num_scales=4)
        else:
            self.isa = None

        # Refinement blocks (简化版)
        self.refinenets = nn.ModuleList([
            self._make_fast_fusion_block(features)
            for _ in range(4)
        ])

        # 深度预测头
        self.depth_heads = nn.ModuleList([
            self._make_depth_head(features)
            for _ in range(4)
        ])

        self.sigmoid = nn.Sigmoid()

    def _make_fast_fusion_block(self, features: int) -> nn.Module:
        """创建快速融合块"""
        return nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def _make_depth_head(self, features: int) -> nn.Module:
        """创建深度预测头"""
        return nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(
        self,
        features: List[Tuple[torch.Tensor, torch.Tensor]],
        patch_h: int,
        patch_w: int
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            features: 编码器输出的特征列表
            patch_h: patch高度
            patch_w: patch宽度

        Returns:
            outputs: 多尺度深度预测
        """
        # 处理输入特征
        out = []
        for i, (feat, _) in enumerate(features):
            x = feat.permute(0, 2, 1).reshape(feat.shape[0], feat.shape[-1], patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        # FPN
        if self.fpn is not None:
            out = self.fpn(out)

        # Conv Neck
        layers = [self.layer_rn[i](out[i]) for i in range(4)]

        # ISA
        if self.isa is not None:
            layers = self.isa(layers)

        # Refinement (自顶向下)
        path_4 = self.refinenets[3](layers[3])
        path_4 = F.interpolate(path_4, size=layers[2].shape[2:], mode='bilinear', align_corners=True)

        path_3 = self.refinenets[2](path_4 + layers[2])
        path_3 = F.interpolate(path_3, size=layers[1].shape[2:], mode='bilinear', align_corners=True)

        path_2 = self.refinenets[1](path_3 + layers[1])
        path_2 = F.interpolate(path_2, size=layers[0].shape[2:], mode='bilinear', align_corners=True)

        path_1 = self.refinenets[0](path_2 + layers[0])

        # 深度预测
        outputs = {
            ("disp", 0): self.sigmoid(self.depth_heads[0](path_1)),
            ("disp", 1): self.sigmoid(self.depth_heads[1](path_2)),
            ("disp", 2): self.sigmoid(self.depth_heads[2](path_3)),
            ("disp", 3): self.sigmoid(self.depth_heads[3](path_4))
        }

        return outputs


class FastViTEncoder(nn.Module):
    """
    高速ViT编码器

    集成Token剪枝功能
    """
    def __init__(
        self,
        base_encoder: nn.Module,
        use_token_pruning: bool = True,
        pruning_layers: List[int] = [4, 8],
        keep_ratios: List[float] = [0.8, 0.6]
    ):
        super().__init__()
        self.encoder = base_encoder
        self.use_token_pruning = use_token_pruning

        if use_token_pruning:
            embed_dim = base_encoder.embed_dim if hasattr(base_encoder, 'embed_dim') else 768
            self.pruning_layers = pruning_layers
            self.keep_ratios = keep_ratios
            self.importance_estimators = nn.ModuleDict({
                str(layer): TokenImportanceEstimator(embed_dim, use_learnable=True)
                for layer in pruning_layers
            })

    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = True
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播

        Args:
            x: 输入图像
            return_intermediate: 是否返回中间特征

        Returns:
            features: 中间层特征列表
        """
        # Patch embedding
        x = self.encoder.prepare_tokens_with_masks(x)

        intermediate_layers = [2, 5, 8, 11]
        outputs = []

        for i, blk in enumerate(self.encoder.blocks):
            x = blk(x)

            # Token剪枝
            if self.use_token_pruning and i in self.pruning_layers:
                layer_idx = self.pruning_layers.index(i)
                keep_ratio = self.keep_ratios[layer_idx]

                if keep_ratio < 1.0:
                    B, N, C = x.shape
                    importance = self.importance_estimators[str(i)](x)
                    importance[:, 0] = float('inf')  # 保留CLS token

                    num_keep = max(int((N - 1) * keep_ratio) + 1, 2)
                    _, keep_indices = torch.topk(importance, num_keep, dim=1)
                    keep_indices = keep_indices.sort(dim=1)[0]

                    batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, num_keep)
                    x = x[batch_indices, keep_indices]

            # 收集中间特征
            if i in intermediate_layers:
                x_norm = self.encoder.norm(x)
                cls_token = x_norm[:, 0]
                patch_tokens = x_norm[:, 1:] if self.encoder.include_cls_token else x_norm
                outputs.append((patch_tokens, cls_token))

        return outputs


class FastEndoDAC(nn.Module):
    """
    高速EndoDAC模型

    整合所有速度优化:
    1. LoRA权重融合
    2. Token剪枝
    3. 高效FPN
    4. 高效ISA
    """
    def __init__(
        self,
        backbone_size: str = "base",
        image_shape: Tuple[int, int] = (224, 280),
        use_token_pruning: bool = True,
        use_efficient_fpn: bool = True,
        use_efficient_isa: bool = True,
        pruning_layers: List[int] = [4, 8],
        keep_ratios: List[float] = [0.8, 0.6],
        fpn_type: str = 'efficient'
    ):
        super().__init__()

        self.image_shape = image_shape
        self.use_token_pruning = use_token_pruning

        # 配置
        self.embedding_dims = {"small": 384, "base": 768}
        self.depth_head_features = {"small": 64, "base": 128}
        self.depth_head_out_channels = {
            "small": [48, 96, 192, 384],
            "base": [96, 192, 384, 768]
        }

        embed_dim = self.embedding_dims[backbone_size]
        features = self.depth_head_features[backbone_size]
        out_channels = self.depth_head_out_channels[backbone_size]

        # 编码器 (需要从外部加载)
        self.encoder = None
        self.fast_encoder = None

        # 解码器
        self.depth_head = FastDPTHead(
            in_channels=embed_dim,
            features=features,
            out_channels=out_channels,
            use_efficient_fpn=use_efficient_fpn,
            use_efficient_isa=use_efficient_isa,
            fpn_type=fpn_type
        )

        # 存储配置
        self.config = {
            'backbone_size': backbone_size,
            'use_token_pruning': use_token_pruning,
            'use_efficient_fpn': use_efficient_fpn,
            'use_efficient_isa': use_efficient_isa,
            'pruning_layers': pruning_layers,
            'keep_ratios': keep_ratios
        }

    def set_encoder(self, encoder: nn.Module):
        """设置编码器"""
        self.encoder = encoder
        if self.use_token_pruning:
            self.fast_encoder = FastViTEncoder(
                encoder,
                use_token_pruning=True,
                pruning_layers=self.config['pruning_layers'],
                keep_ratios=self.config['keep_ratios']
            )

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            pixel_values: 输入图像 [B, 3, H, W]

        Returns:
            outputs: 多尺度深度预测
        """
        # 调整输入尺寸
        pixel_values = F.interpolate(
            pixel_values,
            size=self.image_shape,
            mode='bilinear',
            align_corners=True
        )

        h, w = pixel_values.shape[-2:]
        patch_h, patch_w = h // 14, w // 14

        # 编码
        if self.fast_encoder is not None:
            features = self.fast_encoder(pixel_values)
        elif self.encoder is not None:
            features = self.encoder.get_intermediate_layers(
                pixel_values, 4, return_class_token=True
            )
        else:
            raise RuntimeError("Encoder not set. Call set_encoder() first.")

        # 解码
        outputs = self.depth_head(features, patch_h, patch_w)

        return outputs

    @classmethod
    def from_pretrained(
        cls,
        original_model: nn.Module,
        use_token_pruning: bool = True,
        use_efficient_fpn: bool = True,
        use_efficient_isa: bool = True,
        fuse_lora: bool = True
    ) -> 'FastEndoDAC':
        """
        从预训练模型创建FastEndoDAC

        Args:
            original_model: 原始EndoDAC模型
            use_token_pruning: 是否使用Token剪枝
            use_efficient_fpn: 是否使用高效FPN
            use_efficient_isa: 是否使用高效ISA
            fuse_lora: 是否融合LoRA权重

        Returns:
            fast_model: 优化后的模型
        """
        # 获取配置
        backbone_size = original_model.backbone_size
        image_shape = original_model.image_shape

        # 创建快速模型
        fast_model = cls(
            backbone_size=backbone_size,
            image_shape=image_shape,
            use_token_pruning=use_token_pruning,
            use_efficient_fpn=use_efficient_fpn,
            use_efficient_isa=use_efficient_isa
        )

        # 复制编码器
        fast_model.set_encoder(original_model.encoder)

        # 融合LoRA权重
        if fuse_lora:
            fuse_lora_weights(fast_model.encoder)

        # 复制解码器权重 (需要适配)
        fast_model._copy_decoder_weights(original_model.depth_head)

        return fast_model

    def _copy_decoder_weights(self, original_head: nn.Module):
        """复制解码器权重"""
        # 复制投影层
        for i, proj in enumerate(self.depth_head.projects):
            if hasattr(original_head, 'projects') and i < len(original_head.projects):
                proj.load_state_dict(original_head.projects[i].state_dict())

        # 复制调整层
        for i, resize in enumerate(self.depth_head.resize_layers):
            if hasattr(original_head, 'resize_layers') and i < len(original_head.resize_layers):
                if not isinstance(resize, nn.Identity):
                    try:
                        resize.load_state_dict(original_head.resize_layers[i].state_dict())
                    except:
                        pass

        # 复制Conv Neck
        if hasattr(original_head, 'scratch'):
            for i, layer_rn in enumerate(self.depth_head.layer_rn):
                src_name = f'layer{i+1}_rn'
                if hasattr(original_head.scratch, src_name):
                    src_layer = getattr(original_head.scratch, src_name)
                    try:
                        layer_rn.load_state_dict(src_layer.state_dict())
                    except:
                        pass

    def benchmark(
        self,
        input_size: Tuple[int, int, int, int] = (1, 3, 256, 320),
        num_iterations: int = 100,
        warmup: int = 10,
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """
        性能基准测试

        Args:
            input_size: 输入尺寸 (B, C, H, W)
            num_iterations: 测试迭代次数
            warmup: 预热迭代次数
            device: 设备

        Returns:
            results: 性能指标
        """
        self.eval()
        self.to(device)

        dummy_input = torch.randn(input_size, device=device)

        # 预热
        with torch.no_grad():
            for _ in range(warmup):
                _ = self(dummy_input)

        # 同步
        if device == 'cuda':
            torch.cuda.synchronize()

        # 计时
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                _ = self(dummy_input)
                if device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)

        times = times[5:]  # 去除前几次不稳定的结果

        results = {
            'mean_time_ms': sum(times) / len(times) * 1000,
            'std_time_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5 * 1000,
            'fps': len(times) / sum(times),
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000
        }

        return results


def optimize_for_inference(
    model: nn.Module,
    use_lora_fusion: bool = True,
    use_torch_compile: bool = False,
    use_half_precision: bool = False
) -> nn.Module:
    """
    优化模型用于推理

    Args:
        model: 原始模型
        use_lora_fusion: 是否融合LoRA权重
        use_torch_compile: 是否使用torch.compile (PyTorch 2.0+)
        use_half_precision: 是否使用半精度

    Returns:
        optimized_model: 优化后的模型
    """
    model.eval()

    # LoRA权重融合
    if use_lora_fusion:
        fuse_lora_weights(model)

    # 半精度
    if use_half_precision:
        model = model.half()

    # torch.compile
    if use_torch_compile and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')

    return model
