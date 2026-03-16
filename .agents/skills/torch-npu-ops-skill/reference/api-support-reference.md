## Torch / torch_npu API 支持度参考（示例骨架）

> 说明：本文件给出 **参考结构与示例行**，真实项目中可根据需要补充或更新为更完整的算子清单和精确信息。Skill 在使用时，应将这里视为“强参考而非唯一真理”，并结合用户实际运行情况与官方文档进行判断。

### 1. Torch 原生 API（典型示例）

| 类别 | API 名称 | NPU 支持情况（典型官方配套组合） | 关键约束 | 参考文档入口 |
|------|----------|-----------------------------------|----------|--------------|
| 激活/归一化 | `torch.nn.functional.softmax` / `torch.softmax` | 在多数 CANN + torch_npu 配套版本中支持 NPU；通常依赖底层 softmax 算子 | 一般支持 `float16/float32`，需在 NPU 设备上；数值误差较 CPU 稍大 | PyTorch 官方 softmax 文档；torch_npu 框架特性指南 |
| 矩阵运算 | `torch.matmul` | 在主流版本中支持 NPU，对应 CANN 矩阵乘算子 | 对高维张量广播规则需与 PyTorch 保持一致；部分老版本在特定形状上可能性能较差 | PyTorch matmul 文档；CANN BLAS/MatMul 相关说明 |
| 卷积 | `torch.nn.functional.conv2d` | 在大多数版本配套下支持 NPU | 要求输入/权重布局为 NCHW 或相关格式；部分版本下不支持 groups 很大的情形 | PyTorch conv2d 文档；torch_npu 文档中“卷积与 BN 适配”章节 |
| 归一化 | `torch.nn.functional.layer_norm` | 新版 torch_npu 通常通过自定义 `layer_norm_v3/v4` 等算子适配 NPU | 在部分老版本中可能通过降级到 CPU 或只支持部分 shape | PyTorch layer_norm 文档；torch.ops.npu.layer_norm_v3/v4 自定义算子文档 |

### 2. torch_npu 自定义 API（示例）

| API 名称 | 类型 | NPU 支持情况（典型官方配套组合） | 关键约束 | 参考文档入口 |
|----------|------|-----------------------------------|----------|--------------|
| `torch_npu.npu_softmax` | 功能扩展 | 在较新 torch_npu 版本中提供，对应 CANN softmax 算子 | 输入需在 NPU 上；维度/shape 需满足 CANN softmax 限制；常用于性能敏感场景 | 昇腾 Extension for PyTorch 自定义 API 参考（softmax 相关章节） |
| `torch_npu.npu_format_cast` / `npu_format_cast_` | 存储格式转换 | 详见 `awesome-ascend-skills/torch_npu/SKILL.md` 与官方文档 | 需使用 `torch_npu.Format` 或 int 枚举值；仅支持部分格式互转 | 官方自定义 API 参考中的 Format 说明 |
| `torch.ops.npu.layer_norm_v3` | 自定义算子 | 在支持 LayerNormV3 图算子的版本中可用 | 要求 CANN 中存在 `LayerNormV3` 图算子；输入/输出 shape 需满足 infer shape 逻辑 | op-plugin / torch_npu 自定义算子文档 |
| `torch.ops.npu.layer_norm_v4` | 自定义算子 | 新版本 LayerNorm 自定义算子，支持更灵活的 normalized_shape | normalized_shape 需通过 IntArrayRef 传递，optional gamma/beta 处理方式需遵循文档 | 同上 |
| `torch_npu.npu_grouped_matmul` | 分组矩阵乘 / MoE 核心算子 | 在官方推荐的 `CANN 7.x/8.x + PyTorch 2.x + 对应 torch_npu` 组合中支持 NPU，且文档明确支持 `float16/float32/bfloat16` 等多种 dtype | 典型非量化场景下，输入 `x` 与 `weight` 通常为 **List[Tensor]**，每个元素分别形如 `[M_i, K]` 与 `[K, N_i]`；支持量化/伪量化/pertoken 等模式，需要正确配置 `split_item` / `group_list` / `group_type` / `group_list_type` 等参数。bfloat16 场景中，`x`/`weight` 为 bfloat16，`bias` 建议为 float32，可通过 `output_dtype` 控制输出数据类型 | [`torch_npu.npu_grouped_matmul` 官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_grouped_matmul.md) |

> 使用说明：当 Skill 需要回答“某算子是否支持 NPU”时，可先从此表中查找。如果表中没有，视为“未在参考清单中显式标记”，应结合官方文档与实际运行结果综合判断。

---

### 3. 支持度等级说明建议

在扩展本文件时，建议为每个算子设置一个简单的“支持度”标签，供 Skill 输出时使用：

- **Supported**：在推荐版本组合中已验证支持 NPU；
- **Partial**：仅在部分 dtype/shape 或部分版本中支持；
- **Unknown**：未在 reference 中显式标记，需结合实际运行与官方文档；
- **NotSupported**：在已知版本组合中明确不支持 NPU。

Skill 在回答时可将这些内部标签映射为中文描述，如“支持 / 部分支持 / 当前版本疑似不支持 / 未知（需实测）”。

