#!/usr/bin/env python3
"""
单算子 softmax NPU 示例脚本

展示：
- 环境前置检查（torch / torch_npu 版本、NPU 是否可用）
- 最小输入构造（CPU → NPU）
- 在 NPU 上调用 softmax
- 使用 CPU 结果作为基线进行数值校验
"""

import sys


def check_env():
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover - 环境检查分支
        print("[ERROR] 未安装 torch，请先安装 PyTorch。原始错误：", repr(e))
        sys.exit(1)

    try:
        import torch  # type: ignore
        import torch_npu  # noqa: F401  # type: ignore
    except Exception as e:  # pragma: no cover
        print(
            "[ERROR] 未能成功 import torch_npu。"
            "请确认已安装 torch_npu 且 CANN 环境已正确配置（如 source set_env.sh）。"
        )
        print("原始错误：", repr(e))
        sys.exit(1)

    import torch
    import torch_npu  # type: ignore

    print("torch version    :", torch.__version__)
    print("torch_npu version:", getattr(torch_npu, "__version__", "unknown"))

    if not torch.npu.is_available():
        print(
            "[ERROR] torch.npu.is_available() 为 False。"
            "可能原因：未安装 CANN/torch_npu，或未在当前 shell 中执行 source set_env.sh，"
            "或当前机器无可用 NPU。"
        )
        sys.exit(1)

    return torch


def run_single_op_softmax():
    torch = check_env()
    import torch.nn.functional as F

    # 1. 构造最小输入（CPU 上）
    x_cpu = torch.randn(2, 4, dtype=torch.float32)
    print("input (CPU):")
    print(x_cpu)

    # 2. 迁移到 NPU
    device = torch.device("npu")
    x_npu = x_cpu.to(device)

    # 3. 在 NPU 上调用 softmax
    try:
        y_npu = F.softmax(x_npu, dim=-1)
    except Exception as e:  # pragma: no cover
        print("[ERROR] 在 NPU 上调用 softmax 失败：", repr(e))
        print(
            "这通常意味着当前 CANN/PyTorch/torch_npu 版本组合中 softmax 未适配或存在限制。"
        )
        sys.exit(1)

    # 4. 在 CPU 上计算 baseline
    y_cpu = F.softmax(x_cpu, dim=-1)

    # 5. 将 NPU 结果搬回 CPU 并进行比较
    y_npu_cpu = y_npu.to("cpu")
    diff = (y_npu_cpu - y_cpu).abs().max().item()

    print("output (NPU → CPU):")
    print(y_npu_cpu)
    print("output (CPU baseline):")
    print(y_cpu)
    print("max abs diff:", diff)

    rtol, atol = 1e-3, 1e-5
    if torch.allclose(y_npu_cpu, y_cpu, rtol=rtol, atol=atol):
        print(
            f"[OK] softmax NPU 结果在 rtol={rtol}, atol={atol} 范围内与 CPU 一致，"
            "说明在当前版本组合下 softmax 算子正常可用。"
        )
    else:
        print(
            "[WARN] softmax NPU 结果与 CPU baseline 差异较大，请检查：\n"
            "  - 输入 dtype 是否一致（如 CPU 为 float64 / NPU 为 float16）\n"
            "  - 当前 CANN / torch / torch_npu 版本是否为官方推荐组合\n"
            "  - 是否存在已知的数值问题"
        )


if __name__ == "__main__":
    run_single_op_softmax()

