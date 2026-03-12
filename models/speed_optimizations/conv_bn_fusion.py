"""
卷积层融合：推理时Conv+BN参数折叠 (Inference-time Conv-BN Fusion)

核心思想:
- 对于任何卷积层后的批归一化层，在推理前折叠其参数（更新卷积权重和偏置）
- 直接输出标准化结果，减少卷积和BN的单独操作
- 提升单层效率，常用于实时检测等场景

技术原理:
1. BN的归一化公式: y = (x - mean) / sqrt(var + eps) * weight + bias
2. 融合到Conv: 
   - 新权重 = conv_weight * (bn_weight / sqrt(bn_running_var + eps))
   - 新偏置 = bn_bias - bn_weight * bn_running_mean / sqrt(bn_running_var + eps) + conv_bias * (bn_weight / sqrt(bn_running_var + eps))

预期加速: 5-10% (消除BN层的额外计算开销)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    融合单个Conv2d和BatchNorm2d层的参数
    
    Args:
        conv: 卷积层
        bn: 批归一化层
        
    Returns:
        fused_weight: 融合后的权重
        fused_bias: 融合后的偏置
    """
    # 获取BN参数
    bn_weight = bn.weight.data
    bn_bias = bn.bias.data if bn.bias is not None else torch.zeros_like(bn_weight)
    bn_running_mean = bn.running_mean
    bn_running_var = bn.running_var
    bn_eps = bn.eps
    
    # 计算融合系数
    # scale = bn_weight / sqrt(bn_running_var + eps)
    scale = bn_weight / torch.sqrt(bn_running_var + bn_eps)
    
    # 融合权重: conv_weight * scale
    # 需要将scale reshape为 [out_channels, 1, 1, 1] 以匹配conv权重的形状
    conv_weight = conv.weight.data
    scale_shape = [1] * len(conv_weight.shape)
    scale_shape[0] = -1  # out_channels维度
    scale = scale.view(*scale_shape)
    fused_weight = conv_weight * scale
    
    # 融合偏置
    # 如果conv有bias，先计算 conv_bias * scale
    # 然后加上 bn_bias - bn_weight * bn_running_mean / sqrt(bn_running_var + eps)
    if conv.bias is not None:
        conv_bias = conv.bias.data
        fused_bias = conv_bias * scale.squeeze() + bn_bias - bn_weight * bn_running_mean / torch.sqrt(bn_running_var + bn_eps)
    else:
        # 如果conv没有bias，直接使用BN的偏置项
        fused_bias = bn_bias - bn_weight * bn_running_mean / torch.sqrt(bn_running_var + bn_eps)
    
    return fused_weight, fused_bias


def fuse_conv_bn_eval(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    融合Conv+BN并返回新的Conv层（原地修改）
    
    Args:
        conv: 卷积层
        bn: 批归一化层
        
    Returns:
        融合后的卷积层（原地修改）
    """
    # 确保BN处于eval模式（使用running stats）
    if bn.training:
        bn.eval()
    
    # 融合参数
    fused_weight, fused_bias = fuse_conv_bn(conv, bn)
    
    # 更新conv的权重和偏置
    conv.weight.data = fused_weight
    if conv.bias is None:
        conv.bias = nn.Parameter(fused_bias)
    else:
        conv.bias.data = fused_bias
    
    return conv


def find_conv_bn_pairs(module: nn.Module) -> List[Tuple[str, nn.Conv2d, nn.BatchNorm2d]]:
    """
    查找模型中所有Conv+BN的组合
    
    Args:
        module: 模型
        
    Returns:
        List of (name, conv, bn) tuples
    """
    pairs = []
    
    def _find_pairs(name_prefix: str, m: nn.Module):
        prev_conv = None
        prev_conv_name = None
        
        for name, child in m.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            
            if isinstance(child, nn.Conv2d):
                # 保存当前卷积层
                prev_conv = child
                prev_conv_name = full_name
            elif isinstance(child, nn.BatchNorm2d) and prev_conv is not None:
                # 找到Conv+BN组合
                pairs.append((prev_conv_name, prev_conv, child))
                prev_conv = None
                prev_conv_name = None
            else:
                # 递归查找子模块
                _find_pairs(full_name, child)
                # 如果遇到其他层，重置prev_conv
                if not isinstance(child, (nn.Conv2d, nn.BatchNorm2d, nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                    prev_conv = None
                    prev_conv_name = None
    
    _find_pairs("", module)
    return pairs


def fuse_conv_bn_in_sequential(seq: nn.Sequential) -> nn.Sequential:
    """
    融合Sequential中的Conv+BN组合
    
    Args:
        seq: Sequential模块
        
    Returns:
        融合后的Sequential（原地修改）
    """
    i = 0
    while i < len(seq):
        if isinstance(seq[i], nn.Conv2d) and i + 1 < len(seq) and isinstance(seq[i + 1], nn.BatchNorm2d):
            # 找到Conv+BN组合，进行融合
            conv = seq[i]
            bn = seq[i + 1]
            fuse_conv_bn_eval(conv, bn)
            # 移除BN层（用Identity替换，避免改变索引）
            seq[i + 1] = nn.Identity()
            i += 2
        else:
            i += 1
    return seq


def fuse_conv_bn_weights(model: nn.Module, inplace: bool = True) -> nn.Module:
    """
    融合模型中所有Conv+BN组合的权重
    
    Args:
        model: 包含Conv+BN组合的模型
        inplace: 是否原地修改模型
        
    Returns:
        融合后的模型
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)
    
    # 确保模型处于eval模式
    model.eval()
    
    fused_count = 0
    
    # 方法1: 查找Sequential中的Conv+BN组合
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            # 检查Sequential中是否有Conv+BN组合
            for i in range(len(module) - 1):
                if isinstance(module[i], nn.Conv2d) and isinstance(module[i + 1], nn.BatchNorm2d):
                    conv = module[i]
                    bn = module[i + 1]
                    fuse_conv_bn_eval(conv, bn)
                    # 用Identity替换BN层
                    module[i + 1] = nn.Identity()
                    fused_count += 1
    
    # 方法2: 处理自定义模块中的Conv+BN组合（如ResidualConvUnit）
    # 查找所有包含conv和bn属性的模块
    for name, module in model.named_modules():
        # 处理ResidualConvUnit类型的模块
        if hasattr(module, 'conv1') and hasattr(module, 'bn1'):
            if isinstance(module.conv1, nn.Conv2d) and isinstance(module.bn1, nn.BatchNorm2d):
                fuse_conv_bn_eval(module.conv1, module.bn1)
                module.bn1 = nn.Identity()
                fused_count += 1
        
        if hasattr(module, 'conv2') and hasattr(module, 'bn2'):
            if isinstance(module.conv2, nn.Conv2d) and isinstance(module.bn2, nn.BatchNorm2d):
                fuse_conv_bn_eval(module.conv2, module.bn2)
                module.bn2 = nn.Identity()
                fused_count += 1
        
        # 处理其他可能的命名模式
        if hasattr(module, 'conv') and hasattr(module, 'bn'):
            if isinstance(module.conv, nn.Conv2d) and isinstance(module.bn, nn.BatchNorm2d):
                fuse_conv_bn_eval(module.conv, module.bn)
                module.bn = nn.Identity()
                fused_count += 1
    
    # 方法3: 查找相邻的Conv+BN组合（不在Sequential中，也不在自定义模块中）
    # 通过遍历模块树查找相邻的子模块
    def _fuse_adjacent_conv_bn(parent_module, parent_name=""):
        """递归查找并融合相邻的Conv+BN"""
        nonlocal fused_count
        children = list(parent_module.named_children())
        prev_conv_name = None
        prev_conv = None
        
        for child_name, child_module in children:
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            
            if isinstance(child_module, nn.Conv2d):
                prev_conv_name = child_name
                prev_conv = child_module
            elif isinstance(child_module, nn.BatchNorm2d) and prev_conv is not None:
                # 找到相邻的Conv+BN组合
                fuse_conv_bn_eval(prev_conv, child_module)
                setattr(parent_module, child_name, nn.Identity())
                fused_count += 1
                prev_conv = None
                prev_conv_name = None
            else:
                # 递归处理子模块（跳过已经处理过的Sequential）
                if not isinstance(child_module, nn.Sequential):
                    _fuse_adjacent_conv_bn(child_module, full_name)
                # 如果遇到非Conv/BN的其他层，重置prev_conv
                if not isinstance(child_module, (nn.Conv2d, nn.BatchNorm2d, nn.Sequential, nn.ModuleList, nn.ModuleDict, nn.Identity)):
                    prev_conv = None
                    prev_conv_name = None
    
    # 对模型的所有子模块执行融合
    _fuse_adjacent_conv_bn(model)
    
    return model


def replace_bn_with_identity(model: nn.Module, bn_name: str):
    """
    将指定名称的BN层替换为Identity（辅助函数）
    
    Args:
        model: 模型
        bn_name: BN层的完整名称
    """
    parts = bn_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], nn.Identity())


# 示例：处理ResidualConvUnit中的Conv+BN
def fuse_residual_conv_unit(rcu: nn.Module):
    """
    融合ResidualConvUnit中的Conv+BN组合
    
    Args:
        rcu: ResidualConvUnit模块
    """
    if hasattr(rcu, 'conv1') and hasattr(rcu, 'bn1'):
        if isinstance(rcu.conv1, nn.Conv2d) and isinstance(rcu.bn1, nn.BatchNorm2d):
            fuse_conv_bn_eval(rcu.conv1, rcu.bn1)
            rcu.bn1 = nn.Identity()
    
    if hasattr(rcu, 'conv2') and hasattr(rcu, 'bn2'):
        if isinstance(rcu.conv2, nn.Conv2d) and isinstance(rcu.bn2, nn.BatchNorm2d):
            fuse_conv_bn_eval(rcu.conv2, rcu.bn2)
            rcu.bn2 = nn.Identity()

