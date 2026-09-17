# 可选强教师：DeepSeek生成轨迹 → 本地回放 → Qwen LoRA

更新：2026-09-14。

接入后的完整回归结果：56项测试通过（原49项加7项Teacher门禁/缓存/调用预算测试）。

## 1. 与纯本地版的区别

默认入口仍使用本地规则教师，不联网。添加`--teacher deepseek`后，主要合成任务的正确动作轨迹由DeepSeek生成；任务结构、最小干预、独立数据划分和本地验证器保持相同。各组共有的少量通用回放数据仍由规则教师提供，并在报告中注明。

因此，这版可以检验“强教师产生的监督能否通过门禁并用于训练”。它**没有**让DeepSeek自由生成业务规则、开放能力类别或改写用户任务的语义。自由改写需要额外语义验证，不能只靠JSON正确或最终状态正确就保证训练标签可靠。

不要将纯规则教师结果与DeepSeek教师结果混成同一实验组。用相同Teacher、相同预算比较普通合成与反事实合成。

## 2. 密钥使用

优先读取当前终端环境变量`DEEPSEEK_API_KEY`。不要把真实值粘到运行命令里；可用安全输入：

```powershell
$taskCredential = Read-Host "DeepSeek API key" -AsSecureString
$taskCredentialPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskCredential)
try { $env:DEEPSEEK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskCredentialPtr) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskCredentialPtr) }
```

本机已有旧`config.py`中的配置，显式添加`--use-legacy-key`也可以使用它。代码只读取指定字符串常量，不执行配置代码，不将密钥复制到新文件。未指定该标志时不会读取旧配置。

API固定请求官方`https://api.deepseek.com/chat/completions`。日志不保存Authorization头或密钥；请求审计文件仅包含本实验的合成任务、规则和学生失败描述。

## 3. 先单独验证教师（不训练Qwen）

```powershell
cd "C:\Users\123\Desktop\最小反事实实现\mini_retail_cf"
.\.venv\Scripts\python.exe -u -m experiment.teacher --output outputs/experiments/my_deepseek_probe --use-legacy-key --max-calls 3
```

每个能力调用一次，每次两个任务，最多3次请求。生成6个任务的教师轨迹，回放通过后写入`sft.jsonl`和`teacher_episodes.jsonl`。

只查看请求而不调用API：

```powershell
.\.venv\Scripts\python.exe -m experiment.teacher --output outputs/experiments/my_teacher_requests --dry-run
```

已实际完成的探针位于`outputs/experiments/deepseek_probe_20260914`：模型`deepseek-v4-pro`，3次调用，6/6任务通过回放，19条下一动作SFT样本；API返回prompt tokens 2,567、completion tokens 579，总计3,146。此处数字仅对应该探针，不包括另行运行的完整LoRA联调。

## 4. 跑完整强教师联调

```powershell
.\.venv\Scripts\python.exe -u -m experiment.run --profile smoke --teacher deepseek --use-legacy-key --teacher-max-calls 3 --train-pairs 3 --arms targeted_cf --run-dir outputs/experiments/my_deepseek_smoke
```

流程：底座执行发现集 → 聚类分配3个对等预算 → DeepSeek生成6个任务轨迹 → 回放验证 → 加入共享规则回放 → 本地3步LoRA → 底座/adapter独立评估 → 报告。

这轮只验证接线和数据质量，训练3步不足以判断方法效果。

本机已经实际完成上述完整联调，结果位于 [deepseek_lora_smoke_20260914/REPORT.md](outputs/experiments/deepseek_lora_smoke_20260914/REPORT.md)。该轮额外调用3次API，6/6主要训练任务通过回放；19条强教师动作样本加19条共享规则样本进入LoRA，共3次真实优化器更新。主测试、新措辞和无关变量测试均为0/6；这是工程冒烟结果，尚未完成强教师的充分训练对照。

教师探针与完整联调合计6次真实API调用：prompt tokens 6,087、completion tokens 1,176，总计7,263。调用记录分别保存在两次运行各自的`teacher_cache/ledger.json`，不存在重复计算同一请求。

## 5. 更完整的对照实验

默认pilot的三组、60个对等预算、单轮最多需要180次教师请求：

```powershell
.\.venv\Scripts\python.exe -u -m experiment.run --profile pilot --teacher deepseek --use-legacy-key --teacher-max-calls 180 --run-dir outputs/experiments/deepseek_pilot_s42
```

更小的初步对照：

```powershell
.\.venv\Scripts\python.exe -u -m experiment.run --profile pilot --teacher deepseek --use-legacy-key --train-pairs 15 --teacher-max-calls 45 --discovery-pairs 3 --dev-pairs 2 --test-pairs 3 --ood-pairs 2 --invariance-pairs 1 --run-dir outputs/experiments/deepseek_small_s42
```

每个对等单位用一次请求，普通合成组的一次请求同样包含两个独立任务。实际需求约为：

```text
teacher calls = train_pairs × number_of_arms × rounds
```

程序启动时核对所设调用上限是否够用。上限是请求次数而不是金额；费用以DeepSeek账户实际计费为准。每次最大输出2048 tokens，非thinking模式，温度0.2。

可通过`--teacher-model`指定官方可用模型。当前默认模型和JSON/thinking参数依据2026-09-14的官方文档核对。

## 6. 门禁、缓存与失败处理

- Teacher必须返回请求中的全部task_id，不能遗漏、重复或新增任务。
- 每条轨迹1–7步，仅包含学生动作；工具返回和后续用户消息由真实本地环境生成。
- 前置条件、查询顺序、追问、最新目标、其他订单不变、最终状态及结果都必须通过。
- 最小对两侧全部合格才接受；普通组的两个独立任务也按同样的整体门禁处理，避免只保留容易的样本。
- 不合格时停止，不用规则答案悄悄替换强教师答案，也不继续凑够数据量。
- 接受的响应按请求内容哈希缓存；恢复运行复用缓存并重新回放，不重复调用。
- 调用前先记录ledger。超时或失败后不自动重发；已尝试但未接受的同一请求会提示检查日志。若明确决定重试，使用新输出目录；上一次超时请求可能已计费。

`teacher_cache/`中包括：

```text
ledger.json                  调用次数、状态、task_id及API token用量
<hash>.request.json          可审查的合成任务请求，无密钥
<hash>.candidate.json        教师候选动作（包括被门禁拒绝的已解析候选）
<hash>.json                  已通过回放的响应缓存
```

不会把强教师看到的完整任务元数据作为Student的输入。SFT上下文仍由`World`逐步构造，与Student推理时可见的信息一致。

## 7. 版本与结果说明

加入可选教师后源码与配置指纹已经变化。旧的纯本地实测报告保留，不能在旧运行目录直接执行新代码续跑；请使用新目录。旧运行的源码快照仍位于其`source/experiment/`中。

[给导师看的实测说明.md](给导师看的实测说明.md)记录的是接入强教师前的真实纯本地结果：三种策略在主测试上均为15/18，尚未看到反事实的额外优势。新增强教师接口不自动改变该结论，也不能将探针回放通过率当成学生训练后的成功率。

## 8. 官方接口参考

- [DeepSeek当前模型与接口](https://api-docs.deepseek.com/quick_start/pricing/)
- [JSON输出约束](https://api-docs.deepseek.com/guides/json_mode/)
- [关闭thinking的参数](https://api-docs.deepseek.com/guides/thinking_mode/)
