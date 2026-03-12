"""
创新点1: 推理时LoRA权重融合 (Inference-time LoRA Weight Fusion)

核心思想:
- 在训练时使用低秩分解 W = W0 + B·A (或 DV-LoRA: W = W0 + (B⊙V)·(A⊙U))
- 在推理时将低秩矩阵融合到基础权重中: W_fused = W0 + ΔW
- 消除推理时的额外矩阵乘法，实现零额外计算开销

技术贡献:
1. 提出针对DV-LoRA的权重融合策略，处理动态向量U、V的融合
2. 设计可逆融合机制，支持训练/推理模式无缝切换
3. 实现批量融合接口，一键优化整个模型

预期加速: 10-15% (消除24个LoRA层的额外计算)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class DVLinearFusable(nn.Linear):
    """
    可融合的DV-LoRA线性层

    DV-LoRA公式: output = W0·x + ((B⊙V)·(A⊙U))·x * scaling
    融合后: output = W_fused·x, 其中 W_fused = W0 + (B⊙V)·(A⊙U) * scaling

    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        r: LoRA秩
        lora_alpha: LoRA缩放因子
        lora_dropout: Dropout比率
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        lora_alpha: int = 4,
        lora_dropout: float = 0.,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merged = False

        # LoRA参数
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            # DV-LoRA的动态向量
            self.lora_U = nn.Parameter(torch.zeros(r, 1))
            self.lora_V = nn.Parameter(torch.zeros(out_features, 1))

            # 冻结基础权重
            self.weight.requires_grad = False

        # Dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        """初始化LoRA参数"""
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            nn.init.kaiming_uniform_(self.lora_U, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_V, a=math.sqrt(5))

    def compute_delta_weight(self) -> torch.Tensor:
        """
        计算LoRA增量权重: ΔW = (B⊙V)·(A⊙U) * scaling

        Returns:
            delta_weight: 形状为 (out_features, in_features) 的增量权重
        """
        # A⊙U: (r, in_features) ⊙ (r, 1) -> (r, in_features)
        A_scaled = self.lora_A * self.lora_U
        # B⊙V: (out_features, r) ⊙ (out_features, 1) -> (out_features, r)
        B_scaled = self.lora_B * self.lora_V
        # (B⊙V)·(A⊙U): (out_features, r) @ (r, in_features) -> (out_features, in_features)
        delta_weight = B_scaled @ A_scaled * self.scaling
        return delta_weight

    def merge_weights(self):
        """
        融合LoRA权重到基础权重
        W_fused = W0 + ΔW
        """
        if self.r > 0 and not self.merged:
            delta_weight = self.compute_delta_weight()
            self.weight.data += delta_weight
            self.merged = True

    def unmerge_weights(self):
        """
        解除权重融合，恢复原始权重
        W0 = W_fused - ΔW
        """
        if self.r > 0 and self.merged:
            delta_weight = self.compute_delta_weight()
            self.weight.data -= delta_weight
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        融合模式: output = W_fused·x + bias
        非融合模式: output = W0·x + bias + ((B⊙V)·(A⊙U))·x * scaling
        """
        if self.merged or self.r == 0:
            # 融合模式：直接使用融合后的权重
            return F.linear(x, self.weight, self.bias)
        else:
            # 非融合模式：分别计算基础输出和LoRA输出
            base_output = F.linear(x, self.weight, self.bias)
            # LoRA路径: x @ A^T @ B^T (带动态向量缩放)
            lora_output = self.lora_dropout(x) @ (self.lora_A * self.lora_U).T @ (self.lora_B * self.lora_V).T
            return base_output + lora_output * self.scaling

    def train(self, mode: bool = True):
        """切换训练模式时自动解除融合"""
        super().train(mode)
        if mode and self.merged:
            self.unmerge_weights()
        return self

    def eval(self):
        """切换评估模式时自动融合权重"""
        super().eval()
        if not self.merged and self.r > 0:
            self.merge_weights()
        return self


class LoRAFusionMixin:
    """
    LoRA权重融合混入类

    为模型提供批量融合/解除融合LoRA权重的能力
    """

    def fuse_all_lora_weights(self):
        """融合模型中所有LoRA层的权重"""
        for module in self.modules():
            if isinstance(module, DVLinearFusable):
                module.merge_weights()
            elif hasattr(module, 'merge_weights') and callable(module.merge_weights):
                module.merge_weights()

    def unfuse_all_lora_weights(self):
        """解除模型中所有LoRA层的权重融合"""
        for module in self.modules():
            if isinstance(module, DVLinearFusable):
                module.unmerge_weights()
            elif hasattr(module, 'unmerge_weights') and callable(module.unmerge_weights):
                module.unmerge_weights()

    def get_lora_fusion_status(self) -> dict:
        """获取所有LoRA层的融合状态"""
        status = {}
        for name, module in self.named_modules():
            if isinstance(module, DVLinearFusable):
                status[name] = module.merged
        return status


def fuse_lora_weights(model: nn.Module, inplace: bool = True) -> nn.Module:
    """
    融合模型中所有LoRA权重的工具函数

    Args:
        model: 包含LoRA层的模型
        inplace: 是否原地修改模型

    Returns:
        融合后的模型
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    fused_count = 0
    for name, module in model.named_modules():
        # 处理DVLinearFusable
        if isinstance(module, DVLinearFusable):
            module.merge_weights()
            fused_count += 1
        # 处理原始的DVLinear (来自mylora)
        elif hasattr(module, 'lora_A') and hasattr(module, 'lora_U'):
            if hasattr(module, 'r') and module.r > 0 and not getattr(module, 'merged', False):
                # 计算DV-LoRA增量
                scaling = getattr(module, 'scaling', module.lora_alpha / module.r)
                A_scaled = module.lora_A * module.lora_U
                B_scaled = module.lora_B * module.lora_V
                delta_weight = B_scaled @ A_scaled * scaling
                module.weight.data += delta_weight
                module.merged = True
                fused_count += 1
        # 处理标准LoRA Linear
        elif hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and not hasattr(module, 'lora_U'):
            if hasattr(module, 'r') and module.r > 0 and not getattr(module, 'merged', False):
                scaling = getattr(module, 'scaling', module.lora_alpha / module.r)
                delta_weight = module.lora_B @ module.lora_A * scaling
                module.weight.data += delta_weight
                module.merged = True
                fused_count += 1

    return model


def unfuse_lora_weights(model: nn.Module, inplace: bool = True) -> nn.Module:
    """
    解除模型中所有LoRA权重的融合

    Args:
        model: 包含LoRA层的模型
        inplace: 是否原地修改模型

    Returns:
        解除融合后的模型
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    unfused_count = 0
    for name, module in model.named_modules():
        # 处理DVLinearFusable
        if isinstance(module, DVLinearFusable):
            if module.merged:
                module.unmerge_weights()
                unfused_count += 1
        # 处理原始的DVLinear (来自mylora)
        elif hasattr(module, 'lora_A') and hasattr(module, 'lora_U'):
            if hasattr(module, 'r') and module.r > 0 and getattr(module, 'merged', False):
                # 计算DV-LoRA增量并解除
                scaling = getattr(module, 'scaling', module.lora_alpha / module.r)
                A_scaled = module.lora_A * module.lora_U
                B_scaled = module.lora_B * module.lora_V
                delta_weight = B_scaled @ A_scaled * scaling
                module.weight.data -= delta_weight
                module.merged = False
                unfused_count += 1
        # 处理标准LoRA Linear
        elif hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and not hasattr(module, 'lora_U'):
            if hasattr(module, 'r') and module.r > 0 and getattr(module, 'merged', False):
                scaling = getattr(module, 'scaling', module.lora_alpha / module.r)
                delta_weight = module.lora_B @ module.lora_A * scaling
                module.weight.data -= delta_weight
                module.merged = False
                unfused_count += 1

    return model


def convert_to_fusable_lora(model: nn.Module) -> nn.Module:
    """
    将模型中的DVLinear转换为DVLinearFusable

    Args:
        model: 原始模型

    Returns:
        转换后的模型
    """
    from models.backbones.mylora.layers import DVLinear

    for name, module in model.named_modules():
        if isinstance(module, DVLinear):
            # 创建新的可融合层
            fusable = DVLinearFusable(
                in_features=module.in_features,
                out_features=module.out_features,
                r=module.r,
                lora_alpha=module.lora_alpha,
                bias=module.bias is not None
            )
            # 复制权重
            fusable.weight.data = module.weight.data.clone()
            if module.bias is not None:
                fusable.bias.data = module.bias.data.clone()
            fusable.lora_A.data = module.lora_A.data.clone()
            fusable.lora_B.data = module.lora_B.data.clone()
            fusable.lora_U.data = module.lora_U.data.clone()
            fusable.lora_V.data = module.lora_V.data.clone()

            # 替换模块
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            setattr(parent, child_name, fusable)

    return model
