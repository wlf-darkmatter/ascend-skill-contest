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
   - 若用户只是询问“是否支持”，可以快速给出支持度结论后，**主动询问是否需要测试脚本**，或直接附带一个最小脚本写入到`tests`目录中。
2. **收集版本信息（如果可用）**：
   - 若对话中已有版本信息（例如日志、`pip list`、`npu-smi info` 等），提取 `torch.__version__`、`torch_npu.__version__`、CANN 版本，用于后续兼容性提示。
   - 若用户未提供，则尝试运行相关命令获取对应的版本信息 (例如`python -c "import torch_npu; print(torch_npu.__version__)"`)。
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

### 2.2 官方 torch_npu 自定义 API 文档的**使用原则（禁止自行脑补原型）**

在生成 `torch_npu.npu_xxx` 或 `torch.ops.npu.xxx` 的脚本时，**严禁根据“函数原型文字描述”自行推导/改写参数列表**，而是要：

1. **优先定位“官方最小可运行示例”代码块**，并做到：
   - 逐行**原样照抄调用语句**（尤其是函数名、参数个数与顺序、关键字参数名），不得增删参数；
   - 只允许对「张量创建方式 / shape / dtype / 具体数值」做轻微调整以适配当前测试场景；
   - 不根据“参数说明表格”或“函数原型”自行添加任何新参数。
2. 若同一算子在不同 CANN / torch_npu 版本文档中存在差异：
   - 若用户已给出版本信息，则**尽量选择对应版本的官方文档**；
   - 若版本不明确，则在回答中显式声明：**“以下示例基于文档版本 X.Y，其他版本可能有差异”**，但**仍然严格使用该版本文档中的示例代码，不自行改写签名**。
3. 对于 `torch_npu.npu_*` 算子：
   - 使用 [torch_npu 接口列表](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu_list.md) 仅作为**定位文档入口**；
   - 真正用于拼接测试脚本的调用方式，应来源于该页中的**“示例代码”**片段，而不是页头的原型行。
4. 对于 `torch.ops.npu.*` 算子：
   - 若官方文档中也提供了示例，同样**优先逐字照抄示例调用**；
   - 若没有任何官方示例、只有函数原型或经验性描述，则：
     - 生成脚本时仅使用**文档中明确标记为必填的最小参数集合**；
     - 不随意添加文档未提及的可选参数；
     - 在注释中显式添加：`# 注意：该示例基于文档推断，可能因版本差异导致参数不匹配，请以实际报错为准并参考官方最新示例。`

> **强约束**：  
> - **不要**自己重新翻译/重写函数原型来“优化”调用方式；  
> - **不要**因为觉得“示例里少传了某个你认为重要的参数”就擅自加参数；  
> - 如果不确定，就在脚本注释和回答文字中明确“不确定点”，而不是假装确定。

---


## 3. 单算子测试脚本生成详解

> **最高优先级原则**：  
> - **若官方文档有可直接运行的最小示例代码块**，则以该示例为准，做“最小改动包装”；  
> - **只有当官方没有完整可运行示例**时，才退回到通用模板，并且在调用处严格遵循 2.2 中的“不要脑补签名”规则。

### 3.1 情况 A：基于官方最小示例进行包装（推荐）

当官方文档中已经提供了可运行的示例（含张量创建 + 算子调用）时，应按如下方式生成测试脚本：

1. **完整拷贝示例中的 import & 张量构造 & 算子调用代码**；
2. 在外部增加少量包装，例如：
   - 环境检查函数（可选、简单版即可）；
   - 主函数 `main()`；
   - 可选的 CPU 对比和误差打印；
3. 不改变原有的调用行，只在其前后补充逻辑。

示意结构（伪代码示例，仅说明包装方式，不代表具体算子）：

```python
import torch
import torch_npu

def check_env():
    print("Torch:", torch.__version__)
    print("torch_npu:", torch_npu.__version__)
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        raise RuntimeError("NPU 不可用，请先确认 CANN/torch_npu 环境。")

def main():
    check_env()

    # ===== 以下代码块应来自官方示例，调用行逐字保持一致 =====
    # 官方示例中的张量创建与算子调用（可仅对 shape/数据做轻微调整）
    x = torch.randn(2, 3, 4)           # 来自官方示例
    x_npu = x.to("npu")               # 如官方示例已有则照抄，没有则按最小改动补上
    y_npu = torch.ops.npu.xxx(x_npu)  # ★ 调用行必须与官方示例保持相同参数列表
    # ===== 官方示例结束 =====

    # 可选：CPU baseline 对比
    x_cpu = x
    y_cpu = some_cpu_impl(x_cpu)      # 若官方无 CPU 对比，可使用等价 PyTorch 实现
    print("NPU result:", y_npu[:1].cpu())
    print("CPU result:", y_cpu[:1])

if __name__ == "__main__":
    main()
```

> 关键点：**只在外围包一层，核心调用行不做“自认为更合理”的修改。**

### 3.2 情况 B：官方没有完整示例时使用的通用模板

仅当官方文档**没有给出可直接运行的示例代码**时，才使用下面的通用模板作为兜底，并且：

- 调用行仅使用文档中明确给出的必选参数；
- 不新增文档未出现的参数；
- 在脚本注释中标记“不完全确定”的地方，提醒用户以实际错误信息为准。

推荐结构如下：

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
    #   请根据官方文档中给出的函数原型和“必选参数列表”替换下面这一行，
    #   严禁凭感觉添加额外参数。如果文档只有原型而无示例，请在回答文字中说明这一点。
    y_npu = torch.softmax(x_npu, dim=-1)  # 示例：请替换为目标算子调用

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

### 3.3 脚本中嵌入版本兼容性提示

如果用户提供了版本信息且与官方推荐组合不符，应在脚本前添加注释说明风险，例如：

```python
# 注意：当前环境 CANN 7.0, torch 2.0, torch_npu 2.0 与官方推荐组合 (CANN 8.0 + torch 2.3 + torch_npu 2.3) 有差异，
# 该算子可能在当前环境中存在限制或不可用。请根据实际运行结果判断。
```

如果用户未提供版本，则在脚本开头添加通用提示：

```python
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