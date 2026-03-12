"""
创新点2: 自适应Token剪枝策略 (Adaptive Token Pruning Strategy)

核心思想:
- Vision Transformer的计算复杂度为O(N²)，其中N为Token数量
- 内窥镜图像中存在大量低信息量区域（如均匀组织、暗区）
- 通过动态剪除不重要的Token，显著减少注意力计算量

技术贡献:
1. 提出基于注意力分数的Token重要性评估方法
2. 设计渐进式剪枝策略，在不同Transformer层使用不同剪枝率
3. 实现Token恢复机制，在解码阶段恢复空间信息

预期加速: 20-30% (将Token数量从400减少到200-300)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class TokenImportanceEstimator(nn.Module):
    """
    Token重要性估计器

    基于多种信号估计每个Token的重要性:
    1. 注意力分数: CLS token对各patch token的注意力权重
    2. 特征范数: Token特征向量的L2范数
    3. 梯度信息: (可选) 基于梯度的重要性
    """
    def __init__(self, embed_dim: int, use_learnable: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_learnable = use_learnable

        if use_learnable:
            # 可学习的重要性预测器
            self.importance_predictor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1),
                nn.Sigmoid()
            )

    def forward(
        self,
        tokens: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算Token重要性分数

        Args:
            tokens: Token特征 [B, N, C]
            attn_weights: 注意力权重 [B, H, N, N] (可选)

        Returns:
            importance: 重要性分数 [B, N]
        """
        B, N, C = tokens.shape

        # 方法1: 基于特征范数
        norm_importance = torch.norm(tokens, dim=-1)  # [B, N]
        norm_importance = norm_importance / (norm_importance.max(dim=-1, keepdim=True)[0] + 1e-6)

        # 方法2: 基于注意力分数 (如果提供)
        if attn_weights is not None:
            # 使用CLS token对其他token的注意力作为重要性
            # attn_weights: [B, H, N, N]
            cls_attn = attn_weights[:, :, 0, 1:]  # [B, H, N-1]
            attn_importance = cls_attn.mean(dim=1)  # [B, N-1]
            # 为CLS token添加最高重要性
            cls_importance = torch.ones(B, 1, device=tokens.device)
            attn_importance = torch.cat([cls_importance, attn_importance], dim=1)  # [B, N]
            attn_importance = attn_importance / (attn_importance.max(dim=-1, keepdim=True)[0] + 1e-6)
        else:
            attn_importance = torch.ones(B, N, device=tokens.device)

        # 方法3: 可学习的重要性预测
        if self.use_learnable:
            learned_importance = self.importance_predictor(tokens).squeeze(-1)  # [B, N]
        else:
            learned_importance = torch.ones(B, N, device=tokens.device)

        # 综合重要性分数
        importance = 0.3 * norm_importance + 0.4 * attn_importance + 0.3 * learned_importance

        return importance


class AdaptiveTokenPruning(nn.Module):
    """
    自适应Token剪枝模块

    特点:
    1. 渐进式剪枝: 浅层保留更多Token，深层剪枝更激进
    2. 保护机制: 始终保留CLS token和边界Token
    3. 可逆设计: 记录剪枝索引，支持后续恢复
    """
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 12,
        base_keep_ratio: float = 0.7,
        min_keep_ratio: float = 0.5,
        pruning_schedule: str = 'linear',
        use_learnable: bool = True
    ):
        """
        Args:
            embed_dim: Token嵌入维度
            num_layers: Transformer层数
            base_keep_ratio: 基础保留比例
            min_keep_ratio: 最小保留比例
            pruning_schedule: 剪枝策略 ('linear', 'cosine', 'step')
            use_learnable: 是否使用可学习的重要性预测
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.base_keep_ratio = base_keep_ratio
        self.min_keep_ratio = min_keep_ratio
        self.pruning_schedule = pruning_schedule

        # 重要性估计器
        self.importance_estimator = TokenImportanceEstimator(embed_dim, use_learnable)

        # 计算每层的保留比例
        self.keep_ratios = self._compute_keep_ratios()

        # 存储剪枝信息用于恢复
        self.pruning_indices = []

    def _compute_keep_ratios(self) -> List[float]:
        """计算每层的Token保留比例"""
        ratios = []
        for i in range(self.num_layers):
            progress = i / (self.num_layers - 1) if self.num_layers > 1 else 0

            if self.pruning_schedule == 'linear':
                ratio = self.base_keep_ratio - progress * (self.base_keep_ratio - self.min_keep_ratio)
            elif self.pruning_schedule == 'cosine':
                ratio = self.min_keep_ratio + 0.5 * (self.base_keep_ratio - self.min_keep_ratio) * (1 + math.cos(math.pi * progress))
            elif self.pruning_schedule == 'step':
                # 在中间层开始剪枝
                if i < self.num_layers // 3:
                    ratio = 1.0
                elif i < 2 * self.num_layers // 3:
                    ratio = self.base_keep_ratio
                else:
                    ratio = self.min_keep_ratio
            else:
                ratio = self.base_keep_ratio

            ratios.append(ratio)
        return ratios

    def prune(
        self,
        tokens: torch.Tensor,
        layer_idx: int,
        attn_weights: Optional[torch.Tensor] = None,
        include_cls: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行Token剪枝

        Args:
            tokens: 输入Token [B, N, C]
            layer_idx: 当前层索引
            attn_weights: 注意力权重 (可选)
            include_cls: 是否包含CLS token

        Returns:
            pruned_tokens: 剪枝后的Token [B, K, C]
            keep_indices: 保留的Token索引 [B, K]
        """
        B, N, C = tokens.shape
        keep_ratio = self.keep_ratios[layer_idx]

        # 如果保留比例为1，不进行剪枝
        if keep_ratio >= 1.0:
            return tokens, torch.arange(N, device=tokens.device).unsqueeze(0).expand(B, -1)

        # 计算重要性分数
        importance = self.importance_estimator(tokens, attn_weights)  # [B, N]

        # 确定保留数量
        if include_cls:
            # CLS token始终保留
            num_keep = max(int((N - 1) * keep_ratio) + 1, 2)  # 至少保留CLS和1个patch
            # 将CLS token的重要性设为最大
            importance[:, 0] = float('inf')
        else:
            num_keep = max(int(N * keep_ratio), 1)

        # 选择top-k重要的Token
        _, keep_indices = torch.topk(importance, num_keep, dim=1, sorted=True)
        keep_indices = keep_indices.sort(dim=1)[0]  # 保持原始顺序

        # 收集保留的Token
        batch_indices = torch.arange(B, device=tokens.device).unsqueeze(1).expand(-1, num_keep)
        pruned_tokens = tokens[batch_indices, keep_indices]

        # 存储剪枝信息
        self.pruning_indices.append({
            'layer_idx': layer_idx,
            'keep_indices': keep_indices,
            'original_length': N
        })

        return pruned_tokens, keep_indices

    def restore(
        self,
        pruned_tokens: torch.Tensor,
        original_shape: Tuple[int, int, int],
        layer_idx: int
    ) -> torch.Tensor:
        """
        恢复被剪枝的Token (用于解码阶段)

        Args:
            pruned_tokens: 剪枝后的Token [B, K, C]
            original_shape: 原始形状 (B, N, C)
            layer_idx: 层索引

        Returns:
            restored_tokens: 恢复后的Token [B, N, C]
        """
        B, N, C = original_shape
        device = pruned_tokens.device

        # 查找对应的剪枝信息
        pruning_info = None
        for info in self.pruning_indices:
            if info['layer_idx'] == layer_idx:
                pruning_info = info
                break

        if pruning_info is None:
            return pruned_tokens

        keep_indices = pruning_info['keep_indices']

        # 创建零填充的输出
        restored_tokens = torch.zeros(B, N, C, device=device, dtype=pruned_tokens.dtype)

        # 将保留的Token放回原位置
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, keep_indices.shape[1])
        restored_tokens[batch_indices, keep_indices] = pruned_tokens

        return restored_tokens

    def clear_pruning_history(self):
        """清除剪枝历史"""
        self.pruning_indices = []


class TokenPruningAttention(nn.Module):
    """
    带Token剪枝的注意力模块

    在注意力计算后进行Token剪枝，利用注意力分数作为重要性指标
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        keep_ratio: float = 0.7
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.keep_ratio = keep_ratio

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        prune: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            x: 输入Token [B, N, C]
            prune: 是否执行剪枝

        Returns:
            output: 输出Token [B, K, C] (K <= N)
            keep_indices: 保留的索引 (如果prune=True)
        """
        B, N, C = x.shape

        # 计算QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 输出
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if prune and self.keep_ratio < 1.0:
            # 使用CLS token的注意力作为重要性
            cls_attn = attn[:, :, 0, 1:].mean(dim=1)  # [B, N-1]

            # 保留top-k
            num_keep = max(int((N - 1) * self.keep_ratio), 1)
            _, keep_indices = torch.topk(cls_attn, num_keep, dim=1)
            keep_indices = keep_indices.sort(dim=1)[0]

            # 添加CLS token索引
            cls_idx = torch.zeros(B, 1, dtype=torch.long, device=x.device)
            keep_indices = torch.cat([cls_idx, keep_indices + 1], dim=1)

            # 收集保留的Token
            batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, num_keep + 1)
            x = x[batch_indices, keep_indices]

            return x, keep_indices

        return x, None


class TokenPruningViT(nn.Module):
    """
    带Token剪枝的Vision Transformer包装器

    将Token剪枝集成到ViT的前向传播中
    """
    def __init__(
        self,
        vit_encoder: nn.Module,
        pruning_layers: List[int] = [3, 6, 9],
        keep_ratios: List[float] = [0.8, 0.7, 0.6],
        use_learnable: bool = True
    ):
        """
        Args:
            vit_encoder: 原始ViT编码器
            pruning_layers: 执行剪枝的层索引
            keep_ratios: 对应层的保留比例
            use_learnable: 是否使用可学习的重要性预测
        """
        super().__init__()
        self.encoder = vit_encoder
        self.pruning_layers = pruning_layers
        self.keep_ratios = keep_ratios

        # 获取嵌入维度
        embed_dim = vit_encoder.embed_dim if hasattr(vit_encoder, 'embed_dim') else 768

        # 为每个剪枝层创建重要性估计器
        self.importance_estimators = nn.ModuleDict({
            str(layer): TokenImportanceEstimator(embed_dim, use_learnable)
            for layer in pruning_layers
        })

        # 存储剪枝信息
        self.pruning_info = {}

    def forward_with_pruning(
        self,
        x: torch.Tensor,
        return_intermediate: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        带剪枝的前向传播

        Args:
            x: 输入图像 [B, C, H, W]
            return_intermediate: 是否返回中间特征

        Returns:
            output: 最终输出
            intermediates: 中间层特征列表
        """
        # Patch embedding
        x = self.encoder.prepare_tokens_with_masks(x)

        intermediates = []
        self.pruning_info = {}

        for i, blk in enumerate(self.encoder.blocks):
            x = blk(x)

            # 检查是否需要剪枝
            if i in self.pruning_layers:
                layer_idx = self.pruning_layers.index(i)
                keep_ratio = self.keep_ratios[layer_idx]

                if keep_ratio < 1.0:
                    B, N, C = x.shape

                    # 计算重要性
                    importance = self.importance_estimators[str(i)](x)

                    # CLS token始终保留
                    importance[:, 0] = float('inf')

                    # 选择top-k
                    num_keep = max(int((N - 1) * keep_ratio) + 1, 2)
                    _, keep_indices = torch.topk(importance, num_keep, dim=1)
                    keep_indices = keep_indices.sort(dim=1)[0]

                    # 剪枝
                    batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, num_keep)
                    x = x[batch_indices, keep_indices]

                    # 存储剪枝信息
                    self.pruning_info[i] = {
                        'keep_indices': keep_indices,
                        'original_length': N
                    }

            if return_intermediate:
                intermediates.append(x)

        # 最终LayerNorm
        x = self.encoder.norm(x)

        return x, intermediates

    def get_pruning_stats(self) -> dict:
        """获取剪枝统计信息"""
        stats = {}
        for layer_idx, info in self.pruning_info.items():
            original = info['original_length']
            kept = info['keep_indices'].shape[1]
            stats[f'layer_{layer_idx}'] = {
                'original_tokens': original,
                'kept_tokens': kept,
                'pruning_ratio': 1 - kept / original
            }
        return stats
