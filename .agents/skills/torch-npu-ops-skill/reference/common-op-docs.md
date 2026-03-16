## 常见 Torch / torch_npu 算子文档入口与说明骨架

> 本文件给出若干典型算子的说明结构与文档入口示例，供 Skill 在回答“入参含义 / 返回值说明 / 结果校验方式”类问题时参考。

### 1. softmax 系列

- **Torch 原生：**
  - `torch.nn.functional.softmax(input, dim=None, dtype=None)`  
  - `torch.softmax(input, dim, dtype=None)`  
  - 文档要点：
    - `input`：任意维度张量，通常为 `float16/float32`；
    - `dim`：执行 softmax 的维度，若为 None，默认为最后一维；
    - `dtype`：可选，指定输出 dtype。
  - 结果特性：
    - 在指定维度上归一化后，各元素非负，总和约为 1（考虑浮点误差）。
  - NPU 注意事项：
    - 需将张量迁移到 NPU：`x_npu = x.to("npu")`；
    - 部分版本下，仅支持 `float16/float32`，对 `bfloat16` 或其他 dtype 支持有限；
    - NPU 与 CPU 结果存在一定误差，建议使用 `rtol`/`atol` 宽松比较。

- **torch_npu 自定义：**
  - `torch_npu.npu_softmax(input, axis=-1)`（示例形态，具体以实际文档为准）  
  - 文档要点：
    - `input`：必须为 NPU 上张量；
    - `axis`：与 `dim` 类似，表示归一化维度；
  - 场景：
    - 用于在 NPU 上调用底层 CANN softmax，实现更好的性能或支持特定布局。

### 2. matmul / bmm 系列

- `torch.matmul(input, other)`：
  - 支持二维矩阵乘法、批量矩阵乘法以及高维张量的广播规则；
  - NPU 实现通常映射到底层 GEMM / MatMul 算子；
  - 常见约束：
    - 输入/输出 dtype 通常为 `float16/float32`；
    - 对形状特别大的场景，需留意显存与性能。

- `torch.bmm(input, mat2)`：
  - `input` 为 `(b, n, m)`，`mat2` 为 `(b, m, p)`，输出 `(b, n, p)`；
  - 典型应用于 batch 矩阵乘。

### 2.x Grouped MatMul / MoE 系列（torch_npu.npu_grouped_matmul）

- **算子名称**：`torch_npu.npu_grouped_matmul`
- **核心语义**：
  - 对一组 `(x_i, weight_i)` 做分组矩阵乘法：
    \[
    y_i = x_i @ weight_i + bias_i
    \]
  - 输入通常是 **List[Tensor]**，每个元素对应一个 group / expert。
- **典型非量化调用形态（Python 视角，简化）**：
  ```python
  torch_npu.npu_grouped_matmul(
      x_list,      # List[Tensor]，每个形状 [M_i, K]
      weight_list, # List[Tensor]，每个形状 [K, N_i]
      # 其余参数使用默认值：bias=None, scale=None, offset=None,
      # antiquant_scale=None, antiquant_offset=None, per_token_scale=None,
      # group_list=None, split_item=0, group_type=-1, group_list_type=0,
      # act_type=0, output_dtype=None, tuning_config=None
  )
  ```
- **入参概要（与官方文档对齐）**：
  - `x`：
    - 类型：`List[Tensor]` 或 Tensor；
    - 每个张量 2–6 维，典型为 `[M_i, K]` 或 `[B_i, M_i, K]`；
    - dtype：在 Atlas A2/A3 训练/推理产品上支持 `float32/float16/bfloat16` 及若干整型（量化场景）。
  - `weight`：
    - 类型：`List[Tensor]` 或 Tensor；
    - 每个张量 2D 或 3D，典型为 `[K, N_i]`；
    - 当 `x` 为浮点型（包含 bfloat16）时，`weight` 支持多种浮点/量化 dtype（详见官方数据类型约束表）。
  - 量化相关参数（`bias` / `scale` / `offset` / `antiquant_*` / `per_token_scale`）：
    - 类型：List[Tensor] 或 Tensor；
    - 列表长度与 `weight` 列表长度相同；
    - 用于 per-tensor / per-channel / per-token 量化或伪量化场景。
  - 分组与路由相关参数（`group_list` / `group_type` / `group_list_type`）：
    - 控制沿某一轴的分组方式与 group 索引编码方式，常用于 MoE 路由；
    - 具体取值与含义以官方文档为准。
  - `split_item`：
    - 控制输出张量数量：
      - `0`、`1`：输出为多个张量，数量与 `x` 列表长度相同；
      - `2`、`3`：输出为单个张量。
- **bfloat16 场景建议**：
  - `x_list[i].dtype == torch.bfloat16`，`weight_list[i].dtype == torch.bfloat16`；
  - `bias`（若使用）建议采用 `float32` 以提高数值稳定性；
  - 不显式设置 `output_dtype` 时，输出 dtype 与输入 `x` 相同（即 bfloat16），也可以通过 `output_dtype` 显式指定为 `float32` 等。
- **与 MoE 配置（batch / topk / groupNum）的典型映射**：
  - 对于用户给出的 `(batch, topk, groupNum, K)`：
    - 可以将 `batch` 拆分为 `groupNum * tokens_per_group`，每个 group 内 `tokens_per_group` 个 token；
    - `topk` 通常体现在上游 `npu_moe_gating_top_k` 与下游 `npu_grouped_matmul_finalize_routing` 中，用于决定每个 token 选多少个 expert，本算子通常只负责对已 route 好的 `(x_i, weight_i)` 做矩阵乘；
    - 单算子脚本生成时，可采用简单配置：`groupNum` 个 group，每组 `tokens_per_group = batch / groupNum`，`x_i` 形状 `[tokens_per_group, K]`，`weight_i` 形状 `[K, N]`，使用 List[Tensor] 调用 `torch_npu.npu_grouped_matmul`。
- **结果校验建议**：
  1. 在 CPU 上使用 `float32` 构造 baseline：
     ```python
     y_ref_i = x_cpu_i @ w_cpu_i
     ```
  2. 将 NPU 输出搬回 CPU，并转换为 `float32`：
     ```python
     y_npu_cpu_i = y_npu_i.to("cpu", dtype=torch.float32)
     ```
  3. 对每个 group 使用：
     ```python
     torch.allclose(y_npu_cpu_i, y_ref_i, rtol=1e-2, atol=1e-3)
     ```
     并统计 `max_abs_diff` 作为数值误差参考，结合 bfloat16 精度特性判断算子行为是否合理。

### 3. LayerNorm / BatchNorm 系列

- `torch.nn.functional.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5)`：
  - NPU 上常由 `torch.ops.npu.layer_norm_v3` / `layer_norm_v4` 等自定义算子实现；
  - `normalized_shape`：表示进行归一化的最后若干维度形状；
  - `weight` / `bias`：可选缩放与平移参数。

- `torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5)`：
  - 典型用于卷积网络；
  - NPU 上实现可能复用 CANN 对应 BN 算子。

### 4. 卷积系列（Conv）

- `torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)`：
  - 典型 NCHW 输入：`(N, C_in, H, W)`；
  - 权重：`(C_out, C_in/groups, kH, kW)`；
  - NPU 上要求：
    - 一般支持 NCHW 布局，内部可转换为 NC1HWC0 等格式；
    - groups / dilation 等参数在早期版本中可能支持受限。

### 5. 结果校验通用建议

Skill 在回答任意算子的“结果是否正确”时，可遵循以下通用模式：

1. **构造 CPU Baseline**：
   - 若 Torch CPU 端有同名/等价算子：直接用 CPU 调用；
   - 否则使用组合算子在 CPU 构造参考结果。
2. **将 NPU 结果搬回 CPU**：
   - `out_npu_cpu = out_npu.detach().cpu()`；
3. **数值比较**：
   - 使用 `torch.allclose` 或直接计算 `max_abs_diff`：
     - `diff = (out_npu_cpu - out_cpu).abs().max()`；
   - 根据算子敏感度设置合适的 `rtol`/`atol`（softmax 等归一化算子对误差更敏感，可以适当放宽）。
4. **打印关键信息**：
   - 输出最大绝对误差；
   - 若 diff 较大，提示可能原因：
     - dtype 不一致（如 CPU 为 float64，NPU 为 float16）；
     - 算子在当前版本上的实现存在数值差异或已知问题；
     - 输入非常大/非常小导致数值不稳定。

---

### 6. 扩展建议

在实际项目中，可以在本文件中继续补充：

- 更详细的参数表格（字段、类型、默认值、含义）；
- 针对每个算子的“典型报错”与“排查 checklist”；  
- 针对 transformer / attention / moe 等复杂模块的专门小节。

