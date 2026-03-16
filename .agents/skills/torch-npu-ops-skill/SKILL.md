---
name: torch-npu-ops-skill
description: 提供 Torch 与 torch_npu 算子 API 支持度查询、**单算子最小复现用例生成**（优先）、入参与结果结构化说明，以及基于 CANN / PyTorch / torch_npu 版本组合的兼容性判断和风险提示。当用户 @torch-api、提到 Torch NPU 算子、单算子用例、API 支持度或版本不匹配问题时自动应用。
---

# Torch NPU 算子 API Skill

本 Skill 的核心目标是**生成可直接运行的 Torch NPU 单算子测试脚本**，并在此基础上提供 API 支持度、参数说明和版本兼容性等辅助信息。当用户询问某个算子在 NPU 上的使用情况时，AI 应**优先提供最小可执行脚本**，让用户能够快速验证和调试。

---

## 1. 触发时机（WHEN）

当满足以下任一条件时，应主动使用本 Skill：

- 用户在对话中 **@torch-api**。
- 用户提到如下关键词：**“Torch NPU 算子”**、**“torch_npu 算子”**、**“torch.ops.npu”**、**“单算子用例 / 单算子脚本”**、**“API 支持度 / 是否支持 NPU”**、**“CANN/torch_npu 版本不匹配”**。
- 用户提出与算子相关的问题，例如：
  - “`torch.nn.functional.softmax` 在 NPU 上怎么写测试脚本？”
  - “给我一个 `torch_npu.npu_softmax` 的最小示例。”
  - “如何验证 `torch.bmm` 在 NPU 上的结果？”
  - “我的版本组合下，`torch.ops.npu.layer_norm_v3` 是否可用？帮我生成测试脚本。”

**重点**：只要用户提及具体算子，且可能涉及 NPU 执行，即应视为脚本生成请求。

---

## 2. 总体工作流程（WHAT）

### 2.1 顶层决策步骤

1. **解析用户意图**：
   - 用户是否明确给出了**具体算子名称**（如 `torch.softmax`、`torch_npu.npu_softmax`、`torch.ops.npu.*`）？
   - 用户是否直接要求“脚本”、“示例”、“测试用例”？
   - 若用户只是询问“是否支持”，可以快速给出支持度结论后，**主动询问是否需要测试脚本**，或直接附带一个最小脚本。
2. **收集版本信息（如果可用）**：
   - 若对话中已有版本信息（例如日志、`pip list`、`npu-smi info` 等），提取 `torch.__version__`、`torch_npu.__version__`、CANN 版本，用于后续兼容性提示。
   - 若用户未提供，则生成脚本时在注释中说明“请根据你的环境调整版本相关检查”。
3. **核心输出：生成单算子测试脚本**：
   - 根据算子类型（Torch 原生 / torch_npu 自定义 / torch.ops.npu 自定义），选择合适的脚本模板，填充参数和输入。
   - 脚本必须包含：
     - 环境前置检查（`import`、版本打印、NPU 可用性）
     - 最小输入构造（CPU 生成数据，迁移到 NPU）
     - 算子调用（正确的 API 签名）
     - CPU Baseline 构造（用于结果对比）
     - 结果校验（`torch.allclose` 或最大误差打印）
     - 简单的错误处理提示
   - 输出脚本时，使用代码块（```python）包裹，并附上必要的说明（如期望的误差范围、可能的问题）。
4. **提供辅助信息**（如有必要）：
   - 若用户询问参数含义，可在脚本后附加简短说明。
   - 若版本偏离官方推荐组合，在脚本前添加风险提示。
   - 若算子存在特殊限制（如 dtype、shape 要求），在脚本注释中注明。

> **原则**：先给脚本，再补说明；让用户能直接运行。

---

### 2.2 官方 torch_npu 自定义 API 文档的使用

在生成 `torch_npu.npu_xxx` 或 `torch.ops.npu.xxx` 的脚本时，必须严格遵循官方文档中的函数签名和参数要求。为此：

- 对于 `torch_npu.npu_*` 算子，使用 [torch_npu接口列表](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu_list.md) 查找对应文档页，提取准确的参数列表和约束。这里的列表中如果找到对应的算子，根据其子索引列表继续查找网页文档，精准获取准确的 api 文档链接。
- 对于 `torch.ops.npu.*` 算子，若官方文档未提供，则根据已有经验或本 Skill 的 reference 生成合理示例，并在注释中说明“可能因版本差异而变化”。

脚本生成时，**禁止简化签名**，必须与官方 API 一致。

---

## 3. 单算子测试脚本生成详解

### 3.1 脚本通用模板

所有生成的脚本应遵循以下结构（可根据算子特性微调）：

```python
import torch
import torch_npu  # 如果使用 torch_npu 自定义 API 则必须导入

def check_environment():
    """检查环境并打印版本信息"""
    print("Torch version:", torch.__version__)
    if hasattr(torch, 'npu') and torch.npu.is_available():
        print("NPU is available.")
        if 'torch_npu' in sys.modules:
            print("torch_npu version:", torch_npu.__version__)
        else:
            print("torch_npu not imported.")
    else:
        print("NPU not available. Please check CANN installation and environment.")
        exit(1)

def main():
    # 1. 环境检查
    check_environment()

    # 2. 构造输入（在 CPU 上生成，然后移到 NPU）
    #   根据算子需求选择合适的 shape 和 dtype
    x_cpu = torch.randn(2, 3, 4, dtype=torch.float32)  # 示例输入
    x_npu = x_cpu.to("npu")

    # 3. 调用算子（NPU）
    #   请替换为实际算子调用
    y_npu = torch.softmax(x_npu, dim=-1)  # 示例：torch.softmax

    # 4. 构造 CPU Baseline（使用 Torch 原生 API 或组合实现）
    y_cpu = torch.softmax(x_cpu, dim=-1)  # 同一算子 CPU 版本

    # 5. 结果校验
    y_npu_cpu = y_npu.cpu().float()  # 转回 CPU 并保证 dtype 一致
    diff = (y_cpu - y_npu_cpu).abs().max().item()
    print(f"Max absolute difference: {diff:.6f}")

    # 使用 allclose 进行宽松比较
    rtol = 1e-3
    atol = 1e-5
    if torch.allclose(y_cpu, y_npu_cpu, rtol=rtol, atol=atol):
        print("Result check passed.")
    else:
        print("Result check failed. Please check if the operator is supported correctly.")

if __name__ == "__main__":
    main()

```
### 3.2 根据算子类型调整

#### 3.2.1 Torch 原生算子（如 `torch.add`, `torch.bmm`, `torch.nn.functional.layer_norm`）

- 直接使用模板，将算子调用替换为对应的 Torch API。
- CPU Baseline 直接用同一 API 在 CPU 上执行。
- 注意 NPU 上可能的数据类型限制（如仅支持 float16/float32），在生成输入时指定合适的 dtype。

#### 3.2.2 torch_npu 自定义 API（如 `torch_npu.npu_softmax`）

- 必须导入 `torch_npu`。
- 在调用时使用正确的 API 名称和参数（例如 `torch_npu.npu_softmax(x_npu, axis=-1)`）。
- CPU Baseline 可以使用 Torch 原生对应功能（如 `torch.softmax`）作为参考。
- 如果算子没有 CPU 对应版本（如 `npu_format_cast`），则无法做数值对比，可以在脚本中仅打印输出形状或注释说明。

#### 3.2.3 torch.ops.npu 自定义算子（如 `torch.ops.npu.layer_norm_v3`）

- 调用方式：`torch.ops.npu.layer_norm_v3(x_npu, normalized_shape, weight, bias, eps)`
- 需要根据文档构造正确的输入列表（可能涉及 `List[Tensor]` 等）。
- CPU Baseline 可以用组合算子模拟，或者不做对比（仅打印输出），但要说明原因。

#### 3.2.4 处理 `List[Tensor]` 等复杂输入

- 对于接受 Tensor 列表的算子（如分组 MatMul），必须构造列表输入，不能拼接成单个 Tensor。

- 脚本中应使用列表推导式生成多个 Tensor，例如：

  python

  ```
  num_groups = 2
  x_list = [torch.randn(4, 8).to("npu") for _ in range(num_groups)]
  w_list = [torch.randn(8, 16).to("npu") for _ in range(num_groups)]
  out_list = torch.ops.npu.grouped_matmul(x_list, w_list)
  ```



### 3.3 脚本中嵌入版本兼容性提示

如果用户提供了版本信息且与官方推荐组合不符，应在脚本前添加注释说明风险，例如：

python

```
# 注意：当前环境 CANN 7.0, torch 2.0, torch_npu 2.0 与官方推荐组合 (CANN 8.0 + torch 2.3 + torch_npu 2.3) 有差异，
# 该算子可能在当前环境中存在限制或不可用。请根据实际运行结果判断。
```



如果用户未提供版本，则在脚本开头添加通用提示：

python

```
# 请确保你的环境已正确安装 CANN 和 torch_npu，并且版本匹配。
# 运行前请执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 等环境脚本。
```



------

## 4. 辅助功能（当用户明确询问时提供）

### 4.1 API 支持度查询

若用户仅询问“是否支持”而不要求脚本，可先给出简洁结论，然后主动提供测试脚本：

> “在官方推荐版本组合下，`torch.nn.functional.softmax` 支持 NPU。你可以运行以下脚本验证：” [附脚本]

### 4.2 入参/返回值说明

若用户询问参数含义，可在脚本后附加结构化说明，例如：

**参数说明**：

- `input`：张量，...
- `dim`：归一化的维度，...

### 4.3 版本兼容性判断

若用户询问版本兼容性，可先给出对比结论，然后建议运行脚本实际验证，并附脚本。

------

## 5. 如何使用 reference 与 scripts

本 Skill 附带参考文档和示例脚本，供 AI 在生成脚本时参考：

- **API 支持度与文档入口**：[reference/api-support-reference.md](https://reference/api-support-reference.md) —— 可快速查找算子是否支持。
- **版本对应关系**：[reference/version-compat-matrix.md](https://reference/version-compat-matrix.md) —— 用于判断版本匹配度。
- **常见算子文档**：[reference/common-op-docs.md](https://reference/common-op-docs.md) —— 提供参数说明模板。
- **单算子脚本示例**：[scripts/single_op_softmax_example.py](https://scripts/single_op_softmax_example.py) —— 生成脚本的模板基础。

AI 在生成脚本时，应尽量复用示例脚本的结构，并根据具体算子调整输入 shape 和调用方式。

------

## 6. 回答风格与注意事项

- **语言**：优先简体中文，脚本内注释可使用中文或英文，保持清晰。
- **输出脚本**：必须使用 Markdown 代码块（```python），并附简要说明。
- **避免冗长理论**：除非用户明确要求，否则不展开原理性描述。
- **优先提供可运行脚本**：让用户能够“复制-粘贴-运行”是首要目标。
- **错误处理**：脚本中应包含基本的异常捕获，并打印友好提示。
- **与用户互动**：如果用户信息不足，可以请求补充（如版本），但不要因此延迟脚本输出。