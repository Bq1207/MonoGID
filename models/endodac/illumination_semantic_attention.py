"""
光照-语义注意力模块
包含两个并行分支：
1. 光照分支：SE注意力机制，对特征通道的光照敏感性进行加权
2. 语义分支：轻量级MobileNetV3-based语义分割模型，提取血管、黏膜等语义特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEAttention(nn.Module):
    """Squeeze-and-Excitation注意力模块，用于光照分支"""
    def __init__(self, channels, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LightweightSemanticBranch(nn.Module):
    """轻量级语义分割分支，基于MobileNetV3结构"""
    def __init__(self, in_channels, num_classes=2, reduction=4):
        """
        Args:
            in_channels: 输入特征通道数
            num_classes: 语义类别数（如血管、黏膜等）
            reduction: 通道缩减比例
        """
        super(LightweightSemanticBranch, self).__init__()
        
        # 轻量级特征提取
        mid_channels = in_channels // reduction
        
        # 使用深度可分离卷积减少参数量
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pointwise = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 语义特征提取（输出语义特征图）
        self.semantic_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, x):
        # 轻量级特征提取
        out = self.conv1(x)
        out = self.depthwise(out)
        out = self.pointwise(out)
        
        # 语义特征提取
        semantic_feat = self.semantic_conv(out)
        
        return semantic_feat


class IlluminationSemanticAttention(nn.Module):
    """光照-语义注意力模块"""
    def __init__(self, channels, se_reduction=16, semantic_reduction=4):
        """
        Args:
            channels: 输入特征通道数
            se_reduction: SE注意力的通道缩减比例
            semantic_reduction: 语义分支的通道缩减比例
        """
        super(IlluminationSemanticAttention, self).__init__()
        
        # 光照分支：SE注意力
        self.illumination_branch = SEAttention(channels, reduction=se_reduction)
        
        # 语义分支：轻量级语义分割
        self.semantic_branch = LightweightSemanticBranch(channels, reduction=semantic_reduction)
        
        # 通道级融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 输出归一化
        self.output_norm = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, C, H, W]
        Returns:
            enhanced_feat: 增强后的特征 [B, C, H, W]
        """
        # 光照分支：抑制低光照/过曝区域的噪声特征
        illumination_feat = self.illumination_branch(x)
        
        # 语义分支：提取语义特征
        semantic_feat = self.semantic_branch(x)
        
        # 通道级融合
        fused_feat = torch.cat([illumination_feat, semantic_feat], dim=1)
        enhanced_feat = self.fusion_conv(fused_feat)
        
        # 残差连接
        enhanced_feat = enhanced_feat + x
        
        # 输出归一化
        enhanced_feat = self.output_norm(enhanced_feat)
        
        return enhanced_feat

