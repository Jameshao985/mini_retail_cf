# Mini Retail Counterfactual Experiment v0.3

## GitHub代码快照与本地配置

此仓库保存2026-09-17新方案改造前的代码快照，包括已有训练流程、诊断脚本、测试、任务数据和说明文档。文档中的进度说明按各自编写日期保留。

出于凭据保护，本机的`config.py`不提交。首次克隆后，运行旧版入口或测试之前先创建本地配置：

```powershell
Copy-Item config.example.py config.py
```

本地规则教师不需要API密钥。如使用外部教师，请在本机配置`DEEPSEEK_API_KEY`环境变量；不要把密钥、app_key或密码写入源码、文档或提交消息。`config.py`、`.env`及常见私钥文件已经加入`.gitignore`。

虚拟环境、缓存、`outputs/`运行记录和模型权重不在这个代码快照中。请按照下方依赖说明重建运行环境，并在本地配置模型路径。新克隆的配置模板从环境变量读取密钥，使用外部教师时无需`--use-legacy-key`。

> **2026-09-14 第二版四组实验：** 先阅读 [第二版四组实验说明.md](第二版四组实验说明.md)。
> 新版默认运行普通合成、定向普通合成、普通反事实合成、定向反事实合成四组；加入共同协议预热、五类任务、更具体的失败聚类、错误率预算和固定测试题多seed对照。
> 最终版四组 smoke 已用本地 Qwen 和 LoRA 完整跑通，见 [给导师看的第二版进度.md](给导师看的第二版进度.md)。smoke 只验证工程，尚未运行三seed `study`，也尚未观察到反事实额外收益。
> 第一版真实结果仍保留在 [给导师看的实测说明.md](给导师看的实测说明.md)：三种训练策略在18个主测试任务上均为15/18，尚未观察到反事实额外优势。
> 完整旧版工程细节见 [完整流程与运行说明.md](完整流程与运行说明.md)，可选强教师见 [DeepSeek教师运行说明.md](DeepSeek教师运行说明.md)。
> 下方为保留的旧版说明；其中“尚未实现训练”和旧研究路线只适用于旧入口 `runners.*`。

这是论文《反事实能力诊断驱动的可验证多轮数据合成》的最小科研工程实现。它只完成一条基础链路：让 Qwen3-0.6B 在可程序验证的零售工具环境中执行任务，保存完整轨迹，用最小反事实任务对观察行为变化，并建立 failure pool。它不训练模型，也不声称已经完成因果能力归因。

## 1. 这套代码和论文是什么关系

研究方案的核心不是“失败后让 LLM 猜一个能力标签”，而是：失败轨迹 → 能力假设 → 只改变一个相关变量的 Minimal Counterfactual Pair → 重新执行 Student → 观察行为是否按预期变化 → 后续才可能验证能力归因并生成训练数据。

本 MVP 对应论文八阶段中的早期地基：

- 阶段 A：可验证环境、业务规则、工具、状态迁移和 terminal predicate。
- 阶段 B：记录真实 Student 的失败轨迹并形成 failure pool。
- 阶段 D/E 的最小起点：手工构造单因素配对并报告 observed counterfactual behavior。

本项目已经实现确定性的失败聚类、候选能力假设、DeepSeek 请求模板和本地教师样本回放门禁；尚未实现重复稳定性统计、竞争假设诊断、attribution score、能力隔离数据扩展、效用选择或 LoRA/SFT。因此 pair-level flip 只能叫“观察到的行为翻转”，不能直接叫“因果能力归因”。

## 2. 最少需要理解的 Agent 概念

**Environment（环境）** 是 Agent 可以查询和修改的程序世界。本项目的环境是内存中的零售订单系统，每个 Task 开始前都从该任务的 `initial_state` 深拷贝 reset，上一任务绝不会污染下一任务。

**State（状态）** 是环境在某一时刻保存的事实，例如订单 1001 的 `status=pending`。`snapshot()` 用于保存某一步之前/之后的完整状态，`restore()` 可以恢复快照。

**Tool（工具）** 是 Agent 与环境交互的结构化接口。本项目只有 `get_order(order_id)` 和 `cancel_order(order_id)`。工具输入和输出都是 JSON。

**Rule（规则）** 是工具能否执行的业务 policy。规则独立放在 `env/rules.py`：pending/processing 可取消，shipped/cancelled 不可取消。工具实现不自行复制规则，方便后续把规则作为干预变量。

**Verifier（验证器）** 是程序化裁判，不使用另一个 LLM。它检查工具参数、required/forbidden action、状态查询、状态迁移、terminal predicate、回复与终态是否一致，并明确区分：

- `tool_call_attempt`：Agent 是否尝试调用过工具。
- `tool_execution_success`：至少一个工具是否实际成功执行。
- `task_success`：完整任务是否满足 required action、禁用约束、终态、回复一致性和正常终止。

例如 shipped 订单上尝试 `cancel_order` 时，环境会拒绝修改；但 verifier 仍记录 `attempted_forbidden_action=true`。数据库没被改坏不代表 Agent 的决策正确。

**Task（任务）** 是一次独立实验，包含用户请求、初始状态、干预变量、控制变量、期望行为、required/forbidden action 和 terminal predicate。

**Trajectory（轨迹）** 是一个 episode 的全过程，不只是最后回复。每一步保存 prompt messages、raw model output、parsed action、parse error、工具参数/结果以及环境前后状态。

**Counterfactual Pair（反事实任务对）** 是同一底层场景的两个任务。本地 v0.3 有 10 个 scenario、30 个 pair、60 个 Task。前 10 对保持原 v0.2 的 `pending ↔ shipped` 基线，新增 `pending ↔ cancelled` 与 `processing ↔ shipped` 两组状态比较。每对只主动切换订单状态。

**intervention_variable** 是主动切换的目标变量，这里是 `order.status`。

**controlled_variables** 是必须保持一致的非目标因素，包括 user、order id、item、price、请求文本、工具集合、任务目标和语言风格。pair 内只能主动改变一个变量，否则模型行为差异可能来自语言、价格、工具或难度等混杂因素，无法解释。

`validate_pair_purity()` 会递归比较两个 initial state；如果差异不是唯一的 `orders.<order_id>.status`，或 controlled variables 有任何变化，测试立即失败。

## 3. 为什么先用 ScriptedAgent，再用 Qwen

`ScriptedGoodAgent` 是环境与 verifier 的正控制：它先查询状态，再根据规则取消或拒绝。`BadAgentAlwaysCancel` 和 `BadAgentChecksButIgnores` 是负控制，用来确认 verifier 能分别抓住“完全没查状态”和“查了但忽略结果”。在这些控制 Agent 和普通 pytest 全部通过之前，不应把任何 Qwen 失败解释成模型能力问题。

Qwen3-0.6B 在这里仅是 MVP Student：它足够小，可以本地离线运行，并能暴露 JSON following、tool calling、参数绑定、规则理解和前置条件决策等真实失败。它不是论文最终建议的双模型/多 seed 完整配置。

当前暂时不做 LoRA，因为必须先确认环境、verifier、轨迹、failure pool 和反事实配对可信。若 verifier 本身错误，后续训练只会放大错误标签。

## 4. 为什么 Qwen 失败不能直接证明缺少前置条件检查

一个失败 episode 可能有多种竞争解释：

- `tool_schema_following`：不会按要求输出 JSON。
- JSON formatting：输出截断、代码块或非法 JSON。
- parameter binding：order_id 或参数类型错误。
- rule understanding：查询了状态，但没有正确理解 policy。
- response grounding：动作正确，但最终回复与真实终态矛盾。
- precondition checking：没有在写操作前查询状态，或明知前置条件不满足仍尝试写操作。

因此先运行 sanity check。只有基础 JSON、action、order_id 和 `get_order → respond` 基本通过，后续失败才有资格进入前置条件能力分析；即使如此，正式能力归因仍需要多语言 realization、多 seed、第二个 Student 和竞争假设实验。

## 5. Rule visibility：provided 与 hidden

默认 `provided` 会在 system prompt 中提供一般业务 policy，但不会泄露当前订单状态。模型仍必须主动调用 `get_order`，再把返回状态绑定到动作。这样主要测试“是否主动检查并正确使用状态”，而不是“是否死记电商规则”。

`hidden` 不提供 policy，更接近同时测试规则知识与前置条件检查。两个设置回答的问题不同，结果不能混在一起比较。

## 6. 本地模型路径与严格离线加载

默认缓存根目录：

```text
D:\hf_cache\hub\models--Qwen--Qwen3-0.6B
```

`utils/model_path.py` 会检查其 `snapshots` 子目录，只接受同时包含 `config.json`、tokenizer 文件和模型权重的完整 snapshot。自动发现失败会明确报错；代码不会回退到在线下载。Tokenizer 和模型均使用 `local_files_only=True`，运行实验时还建议设置 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`。

可以用环境变量或 CLI 覆盖路径。PowerShell：

```powershell
$env:MODEL_PATH = "D:\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\<commit_hash>"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

CMD：

```bat
set MODEL_PATH=D:\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\<commit_hash>
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
```

也可直接传参：

```powershell
python -m runners.run_qwen --model-path "D:\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\<commit_hash>" --limit 2
```

## 7. 安装与运行顺序

进入项目目录：

```powershell
cd "C:\Users\123\Desktop\最小反事实实现\mini_retail_cf"
python --version
python -c "import torch, transformers; print(torch.__version__); print(transformers.__version__); print(torch.cuda.is_available())"
```

仅在缺少依赖时安装最小 requirements；不要安装 LangChain、LangGraph、bitsandbytes、PEFT、TRL 或数据库：

```powershell
python -m pip install -r requirements.txt
```

先运行快速测试：

```powershell
python -m pytest -q
```

运行三个控制 Agent：

```powershell
python -m runners.run_batch --agent scripted
python -m runners.run_batch --agent always_cancel
python -m runners.run_batch --agent checks_but_ignores
```

Qwen sanity check：

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m runners.run_qwen --sanity-check
```

先跑 pair_001 的两个任务：

```powershell
python -m runners.run_qwen --limit 2
```

确认 trajectory 后跑全部 60 个任务：

```powershell
python -m runners.run_qwen
```

隐藏规则实验是单独配置：

```powershell
python -m runners.run_qwen --rule-visibility hidden --limit 2
```

CMD 中长命令可用 `^` 换行：

```bat
python -m runners.run_qwen ^
  --model-path "D:\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\<commit_hash>" ^
  --limit 2
```

## 8. 多步 Qwen Agent loop

模型每次只能输出一个动作 JSON。程序把用户请求与历史工具结果交给 Qwen；parser 只去除代码块/外围文本并提取第一个合法 JSON object，不会把错误动作、order_id 或业务决策改成正确答案。解析失败会原样记录并终止为 `parse_error`。

合法工具动作会照常执行，工具结果加入 history 后再次调用 Qwen。`max_steps=6`；如果模型持续调用工具而不回复，episode 以 `max_steps_exceeded` 失败。

默认生成设置为 seed 42、`do_sample=False`、`max_new_tokens=256`。Qwen3 chat template 若支持 `enable_thinking=False` 就关闭 thinking；旧接口不支持时使用兼容 fallback。模型自动优先 CUDA，否则使用 CPU，不做 4-bit/8-bit 量化。

## 9. 如何阅读输出

完整轨迹位于：

```text
outputs/trajectories/<label>_trajectories.jsonl
```

JSONL 每行是一个完整 episode。建议按以下顺序查看：`task_id` → `initial_state` → `steps[].raw_model_output` → `parsed_action` → `tool_result` → state before/after → `final_state` → `verifier_result`。

failure pool 位于：

```text
outputs/failures/<label>_failures.jsonl
```

只要 `task_success=false` 或存在 `violation_types` 就会进入。failure type 是可观察错误，不等于已完成的能力诊断：

- `parse_error`
- `wrong_tool`
- `wrong_argument`
- `missing_state_check`
- `forbidden_action_attempt`
- `illegal_state_transition`
- `max_steps_exceeded`
- `wrong_terminal_state`
- `other`

pair-level 结果和总表位于：

```text
outputs/summaries/<label>_pair_results.json
outputs/summaries/<label>_summary.json
```

每次运行还会按唯一 `experiment_id` 保存一份完整副本，避免不同模型或配置相互覆盖：

```text
outputs/runs/<experiment_id>/
├── manifest.json
├── trajectories.jsonl
├── failures.jsonl
├── pair_results.json
└── summary.json
```

可以用确定性的聚合器把大规模失败池压缩成可审查的失败簇：

```powershell
python -m runners.analyze_failures `
  --input outputs/trajectories.jsonl `
  --output outputs/analysis/failure_report.json
```

该报告只根据 verifier 的可观察字段聚类，不替代后续的能力假设和反事实验证。

在失败簇基础上，可以生成结构化的候选能力假设：

```powershell
python -m runners.propose_hypotheses `
  --input outputs/analysis/qwen_failure_report.json `
  --output outputs/analysis/candidate_hypotheses.json
```

候选假设只用于组织后续实验，必须重新构造纯反事实对并运行 Student 后，才能标记为支持或拒绝；脚本不会把聚类结果直接当成因果能力归因。

开发集之外的独立验证集可以这样生成和运行正控制：

```powershell
python data/generate_validation_tasks.py
python -m runners.run_batch `
  --agent scripted `
  --tasks-path data/validation_tasks.json `
  --label scripted_validation
```

Qwen 运行时使用同样的 `--tasks-path data/validation_tasks.json`，并指定一个独立的 `--label`；验证集不能用于生成训练数据。

如果验证集把“执行取消”和“咨询能否取消”混在一起，需要用命令式复制集隔离请求意图：

```powershell
python data/generate_replication_tasks.py
python -m runners.run_batch `
  --agent scripted `
  --tasks-path data/replication_tasks.json `
  --label scripted_replication
```

该复制集使用新订单号和新的措辞，但每条请求都明确要求执行取消；它适合在进入教师数据生成前，检验状态干预本身是否能稳定复现。

### 9.1 教师模型数据请求（DeepSeek 模板）

复制集确认可复现后，先把失败轨迹聚合为报告，再生成候选能力假设：

```powershell
python -m runners.analyze_failures `
  --input outputs/runs/qwen-<experiment_id>/trajectories.jsonl `
  --output outputs/analysis/qwen_replication_failure_report.json
python -m runners.propose_hypotheses `
  --input outputs/analysis/qwen_replication_failure_report.json `
  --output outputs/analysis/qwen_replication_hypotheses.json
```

先不联网检查请求内容：

```powershell
python -m runners.generate_teacher_data `
  --hypotheses outputs/analysis/qwen_replication_hypotheses.json `
  --trajectories outputs/runs/qwen-<experiment_id>/trajectories.jsonl `
  --dry-run
```

该命令只写 `outputs/teacher/deepseek_requests.jsonl`，不会读取 Key，也不会调用 API。确认请求内容后，安装 `openai`，然后把真实 Key 填入 `config.py` 的 `DEEPSEEK_API_KEY`。也可以继续使用环境变量；环境变量优先级更高。无论哪种方式，都不要提交真实 Key 或把它放进截图：

```powershell
python -m pip install -r requirements-next.txt
# 方案 A：编辑 config.py，把 DEEPSEEK_API_KEY = "" 改成真实 Key
# 方案 B：仅当前 PowerShell 会话临时覆盖配置
# $env:DEEPSEEK_API_KEY = "把真实Key临时粘贴到这里"
python -m runners.generate_teacher_data `
  --hypotheses outputs/analysis/qwen_replication_hypotheses.json `
  --trajectories outputs/runs/qwen-<experiment_id>/trajectories.jsonl `
  --model deepseek-v4-pro `
  --output outputs/teacher/deepseek_candidates.jsonl
```

脚本会把每个候选假设和代表性失败轨迹组成一个请求，要求教师模型只返回纠错对话 JSON；程序检查每个 assistant 消息是否为合法 action、是否先 `get_order` 再 `cancel_order`，不合格响应不会写入候选训练集。输出记录带有 `needs_student_verification=true`，所以它们仍不是可直接训练的金标准。真实训练前，要把这些候选样本重新放回环境和 verifier，通过后再进入 LoRA/SFT。

拿到候选 JSONL 后先做本地回放门禁：

```powershell
python -m runners.validate_teacher_data `
  --candidates outputs/teacher/deepseek_candidates.jsonl `
  --trajectories outputs/runs/qwen-<experiment_id>/trajectories.jsonl
```

只有 `outputs/teacher/accepted_candidates.jsonl` 中的样本同时满足状态迁移、工具调用和最终回复检查，才有资格进入下一步训练；`validation_report.json` 保留每条样本被接受或拒绝的原因。

默认 `provided` 的 60-task 正式运行还会同步写出四个 canonical 交付文件：`outputs/trajectories.jsonl`、`outputs/failures.jsonl`、`outputs/pair_results.json` 和 `outputs/summary.json`。

每对会被记录为 PASS/PASS、PASS/FAIL、FAIL/PASS 或 FAIL/FAIL。`success_flip` 表示两侧 task success 不同，只是 observed counterfactual behavior，不能单独证明因果归因。

## 10. 项目结构

```text
mini_retail_cf/
├── config.py                 # 离线模型、seed、步数和输出默认值
├── data/
│   ├── database.json         # 四种状态的示例数据库
│   ├── generate_tasks.py     # 确定性生成 30 个任务对
│   ├── generate_validation_tasks.py # 独立验证集生成器
│   ├── generate_replication_tasks.py # 命令式复制集生成器
│   └── tasks.json            # 实验实际读取的 60 个静态任务
├── env/
│   ├── environment.py        # reset/snapshot/restore 与状态读写
│   ├── rules.py              # 独立取消规则
│   ├── tools.py              # 结构化 get/cancel 工具
│   └── verifier.py           # 硬验证与 pair purity
├── agents/
│   ├── base_agent.py         # 多步 loop 与逐步轨迹
│   ├── scripted_agent.py     # 正控制
│   ├── bad_agents.py         # 两个负控制
│   └── qwen_agent.py         # 本地离线 Qwen Student
├── runners/
│   ├── common.py             # batch、summary、failure pool
│   ├── analyze_failures.py   # 失败池确定性聚类
│   ├── propose_hypotheses.py # 候选能力假设生成
│   ├── generate_teacher_data.py # DeepSeek 请求与教师 JSONL 校验模板
│   └── validate_teacher_data.py # 候选样本本地回放与 verifier 门禁
│   ├── run_single.py         # 单任务检查
│   ├── run_batch.py          # 控制 Agent CLI
│   └── run_qwen.py           # sanity 与真实 Qwen CLI
├── utils/
│   ├── model_path.py         # snapshot 自动发现与完整性检查
│   ├── action_parser.py      # 保守 JSON action parser
│   └── trajectory_logger.py  # JSON/JSONL 输出
├── outputs/                  # trajectories、failures、summaries
└── tests/                    # 快速程序测试；不加载真实 Qwen
```

相对推荐结构只增加了确定性任务生成器、失败分析/假设脚本和教师数据门禁：前者防止静态任务手工复制时产生 pair 漂移，后者把轨迹、候选假设和可验证样本串成可审计链路。所有脚本都很小，不引入 Agent 框架。

## 11. 当前科研结论边界

这套 MVP 可以支持：环境可程序验证；Qwen3-0.6B 能作为真实 Student；可以保存多步失败轨迹；可以构造并自动检查单因素任务对；可以观察状态干预后的行为差异。

它不能支持：能力因果归因已经成立；本方法优于 TRACE；合成数据更适合训练；LoRA 会提升目标能力；能力可跨模型、跨 seed 或跨领域泛化。下一阶段应先增加 capability hypothesis、竞争假设诊断、重复 counterfactual rollout 和 attribution score，再考虑数据扩展与训练。
