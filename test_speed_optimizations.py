"""
速度优化测试脚本

测试四个创新点的加速效果:
1. LoRA权重融合
2. Token剪枝
3. 高效FPN
4. 高效ISA

使用方法:
    python test_speed_optimizations.py --model_path <path_to_model>
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def benchmark_model(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 256, 320),
    num_iterations: int = 100,
    warmup: int = 10,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    模型性能基准测试

    Args:
        model: 待测试模型
        input_size: 输入尺寸
        num_iterations: 测试迭代次数
        warmup: 预热次数
        device: 设备

    Returns:
        results: 性能指标
    """
    model.eval()
    model.to(device)

    dummy_input = torch.randn(input_size, device=device)

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if device == 'cuda':
        torch.cuda.synchronize()

    # 计时
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    times = np.array(times[5:])  # 去除前几次

    return {
        'mean_ms': times.mean() * 1000,
        'std_ms': times.std() * 1000,
        'min_ms': times.min() * 1000,
        'max_ms': times.max() * 1000,
        'fps': 1.0 / times.mean()
    }


def test_lora_fusion():
    """测试LoRA权重融合"""
    print("\n" + "="*60)
    print("测试创新点1: LoRA权重融合")
    print("="*60)

    from models.speed_optimizations.lora_fusion import DVLinearFusable, fuse_lora_weights

    # 创建测试层
    layer = DVLinearFusable(768, 3072, r=4)
    layer.cuda()
    layer.eval()

    x = torch.randn(1, 400, 768, device='cuda')

    # 未融合
    with torch.no_grad():
        times_unfused = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = layer(x)
            torch.cuda.synchronize()
            times_unfused.append(time.perf_counter() - start)

    # 融合
    layer.merge_weights()

    with torch.no_grad():
        times_fused = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = layer(x)
            torch.cuda.synchronize()
            times_fused.append(time.perf_counter() - start)

    unfused_mean = np.mean(times_unfused[10:]) * 1000
    fused_mean = np.mean(times_fused[10:]) * 1000
    speedup = unfused_mean / fused_mean

    print(f"未融合延迟: {unfused_mean:.3f}ms")
    print(f"融合后延迟: {fused_mean:.3f}ms")
    print(f"加速比: {speedup:.2f}x")

    return speedup


def test_token_pruning():
    """测试Token剪枝"""
    print("\n" + "="*60)
    print("测试创新点2: 自适应Token剪枝")
    print("="*60)

    from models.speed_optimizations.token_pruning import AdaptiveTokenPruning

    pruner = AdaptiveTokenPruning(
        embed_dim=768,
        num_layers=12,
        base_keep_ratio=0.8,
        min_keep_ratio=0.5
    )
    pruner.cuda()

    # 模拟Token
    tokens = torch.randn(1, 401, 768, device='cuda')  # 400 patches + 1 CLS

    # 测试不同层的剪枝
    print("\n各层剪枝情况:")
    for layer_idx in [0, 3, 6, 9, 11]:
        pruned, indices = pruner.prune(tokens, layer_idx)
        keep_ratio = pruned.shape[1] / tokens.shape[1]
        print(f"  Layer {layer_idx}: {tokens.shape[1]} -> {pruned.shape[1]} tokens (保留 {keep_ratio*100:.1f}%)")

    # 计算理论加速
    # 注意力复杂度 O(N²)
    original_cost = 401 ** 2
    pruned_costs = []
    current_tokens = 401
    for layer_idx in range(12):
        pruned, _ = pruner.prune(torch.randn(1, current_tokens, 768, device='cuda'), layer_idx)
        current_tokens = pruned.shape[1]
        pruned_costs.append(current_tokens ** 2)

    total_original = original_cost * 12
    total_pruned = sum(pruned_costs)
    theoretical_speedup = total_original / total_pruned

    print(f"\n理论注意力计算量减少: {(1 - total_pruned/total_original)*100:.1f}%")
    print(f"理论加速比: {theoretical_speedup:.2f}x")

    return theoretical_speedup


def test_efficient_fpn():
    """测试高效FPN"""
    print("\n" + "="*60)
    print("测试创新点3: 轻量级动态FPN")
    print("="*60)

    from models.endodac.bifpn import ChannelPreservingEnhancedBiFPN
    from models.speed_optimizations.efficient_fpn import ChannelPreservingEfficientFPN

    channels = [96, 192, 384, 768]

    # 原始BiFPN
    original_fpn = ChannelPreservingEnhancedBiFPN(channels, unified_channels=256, num_bifpn_blocks=2)
    original_fpn.cuda()
    original_fpn.eval()

    # 高效FPN
    efficient_fpn = ChannelPreservingEfficientFPN(channels, fpn_type='efficient')
    efficient_fpn.cuda()
    efficient_fpn.eval()

    # 创建测试输入
    features = [
        torch.randn(1, 96, 64, 80, device='cuda'),
        torch.randn(1, 192, 32, 40, device='cuda'),
        torch.randn(1, 384, 16, 20, device='cuda'),
        torch.randn(1, 768, 8, 10, device='cuda')
    ]

    # 测试原始FPN
    with torch.no_grad():
        times_original = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = original_fpn(features)
            torch.cuda.synchronize()
            times_original.append(time.perf_counter() - start)

    # 测试高效FPN
    with torch.no_grad():
        times_efficient = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = efficient_fpn(features)
            torch.cuda.synchronize()
            times_efficient.append(time.perf_counter() - start)

    original_mean = np.mean(times_original[10:]) * 1000
    efficient_mean = np.mean(times_efficient[10:]) * 1000
    speedup = original_mean / efficient_mean

    # 参数量对比
    original_params = sum(p.numel() for p in original_fpn.parameters())
    efficient_params = sum(p.numel() for p in efficient_fpn.parameters())

    print(f"原始FPN延迟: {original_mean:.3f}ms")
    print(f"高效FPN延迟: {efficient_mean:.3f}ms")
    print(f"加速比: {speedup:.2f}x")
    print(f"\n原始FPN参数量: {original_params:,}")
    print(f"高效FPN参数量: {efficient_params:,}")
    print(f"参数减少: {(1 - efficient_params/original_params)*100:.1f}%")

    return speedup


def test_efficient_isa():
    """测试高效ISA"""
    print("\n" + "="*60)
    print("测试创新点4: 高效ISA模块")
    print("="*60)

    from models.endodac.illumination_semantic_attention import IlluminationSemanticAttention
    from models.speed_optimizations.efficient_isa import EfficientISA, SharedISA

    channels = 128

    # 原始ISA (4个独立模块)
    original_isas = nn.ModuleList([
        IlluminationSemanticAttention(channels) for _ in range(4)
    ])
    original_isas.cuda()
    original_isas.eval()

    # 高效ISA (共享参数)
    shared_isa = SharedISA(channels, num_scales=4)
    shared_isa.cuda()
    shared_isa.eval()

    # 创建测试输入
    features = [
        torch.randn(1, 128, 64, 80, device='cuda'),
        torch.randn(1, 128, 32, 40, device='cuda'),
        torch.randn(1, 128, 16, 20, device='cuda'),
        torch.randn(1, 128, 8, 10, device='cuda')
    ]

    # 测试原始ISA
    with torch.no_grad():
        times_original = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = [isa(feat) for isa, feat in zip(original_isas, features)]
            torch.cuda.synchronize()
            times_original.append(time.perf_counter() - start)

    # 测试高效ISA
    with torch.no_grad():
        times_efficient = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = shared_isa(features)
            torch.cuda.synchronize()
            times_efficient.append(time.perf_counter() - start)

    original_mean = np.mean(times_original[10:]) * 1000
    efficient_mean = np.mean(times_efficient[10:]) * 1000
    speedup = original_mean / efficient_mean

    # 参数量对比
    original_params = sum(p.numel() for p in original_isas.parameters())
    efficient_params = sum(p.numel() for p in shared_isa.parameters())

    print(f"原始ISA延迟: {original_mean:.3f}ms")
    print(f"高效ISA延迟: {efficient_mean:.3f}ms")
    print(f"加速比: {speedup:.2f}x")
    print(f"\n原始ISA参数量: {original_params:,}")
    print(f"高效ISA参数量: {efficient_params:,}")
    print(f"参数减少: {(1 - efficient_params/original_params)*100:.1f}%")

    return speedup


def test_full_model(model_path: str = None):
    """测试完整模型优化"""
    print("\n" + "="*60)
    print("测试完整模型优化")
    print("="*60)

    if model_path is None:
        print("未提供模型路径，跳过完整模型测试")
        return None

    from models.endodac.endodac import endodac
    from models.speed_optimizations import optimize_for_inference, fuse_lora_weights

    # 加载模型
    print("加载模型...")
    model = endodac(
        backbone_size="base",
        r=4,
        lora_type="dvlora",
        image_shape=(224, 280),
        use_bifpn=True,
    )

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

    model.cuda()
    model.eval()

    # 测试原始模型
    print("\n测试原始模型...")
    original_results = benchmark_model(model)
    print(f"原始模型 FPS: {original_results['fps']:.2f}")
    print(f"原始模型延迟: {original_results['mean_ms']:.2f}ms")

    # 优化模型
    print("\n优化模型...")
    optimized_model = optimize_for_inference(model, use_lora_fusion=True)

    # 测试优化后模型
    print("\n测试优化后模型...")
    optimized_results = benchmark_model(optimized_model)
    print(f"优化后 FPS: {optimized_results['fps']:.2f}")
    print(f"优化后延迟: {optimized_results['mean_ms']:.2f}ms")

    speedup = original_results['mean_ms'] / optimized_results['mean_ms']
    print(f"\n总体加速比: {speedup:.2f}x")

    return speedup


def main():
    parser = argparse.ArgumentParser(description='速度优化测试')
    parser.add_argument('--model_path', type=str, default=None, help='模型权重路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        args.device = 'cpu'

    print("="*60)
    print("EndoDAC-FPN 速度优化测试")
    print("="*60)

    results = {}

    # 测试各个创新点
    try:
        results['lora_fusion'] = test_lora_fusion()
    except Exception as e:
        print(f"LoRA融合测试失败: {e}")

    try:
        results['token_pruning'] = test_token_pruning()
    except Exception as e:
        print(f"Token剪枝测试失败: {e}")

    try:
        results['efficient_fpn'] = test_efficient_fpn()
    except Exception as e:
        print(f"高效FPN测试失败: {e}")

    try:
        results['efficient_isa'] = test_efficient_isa()
    except Exception as e:
        print(f"高效ISA测试失败: {e}")

    # 测试完整模型
    if args.model_path:
        try:
            results['full_model'] = test_full_model(args.model_path)
        except Exception as e:
            print(f"完整模型测试失败: {e}")

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for name, speedup in results.items():
        if speedup is not None:
            print(f"{name}: {speedup:.2f}x 加速")


if __name__ == '__main__':
    main()
