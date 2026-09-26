# HotpotQA 三智能体 Base 实验实现规格

> 版本：v1.1  
> 用途：供编码 Agent 直接实现第一阶段实验  
> 状态：本文只写入当前已经确认的 Base 设定；扩展实验统一列在“暂不实现”部分。

## 1. 实验目标

本实验建立一个未经通信压缩或协议优化的多智能体基线（free-form MAS baseline）。核心研究问题是：

> 当回答一个问题所需的信息分散在两个互相隔离的 Agent 中，而协调者只看到问题时，三个 Agent 在自由自然语言通信下会产生多少生成 token、进行多少次交互，并能否得到正确答案？

第一阶段只验证这一基础设置能否运行，并测量其自然通信行为。Base 的目的不是主动减少 token，也不是通过 prompt 教模型使用某种高效策略。后续方法再以此为参照，在尽量保持 Answer F1 的前提下降低生成 token。

当前实验可概括为：

```text
3 independent agents
+ star topology
+ private relevant evidence
+ free-form natural-language messages
+ coordinator aggregation
```

## 2. 当前实验范围

| 项目 | 固定设置 |
|---|---|
| 数据集 | HotpotQA validation split |
| 问题数 | 100 个固定问题 |
| 每题运行次数 | 10 次独立采样 |
| 总 trajectory 数 | 1000 |
| 模型 | Gemma-3-1B-IT |
| Agent | Alice、Bob、Celab |
| Agent 关系 | 三个独立 Agent instance |
| 通信方式 | 自由自然语言通信；v1 调度实现为顺序、异步 |
| 最大正常 decision steps | 20 |
| 正常终止标记 | `<FINAL>answer</FINAL>` |
| 主效果指标 | HotpotQA Answer F1 |
| 辅助效果指标 | Answer EM |
| 主效率指标 | Total Generated Tokens (`#GenTok`) |
| 主交互指标 | Decision Steps |
| 失败/失控指标 | Cap Rate、generation-cap rate |

这里的 HotpotQA 实验是一个 **controlled information-asymmetric setting**：只把相关证据分给 Alice 和 Bob，不加入 distractor documents。不要把它描述成标准 HotpotQA full-context/distractor setting。

## 3. Agent 定义与职责

### 3.1 Alice

Alice 是私有证据持有者。v1 使用 Celab-driven 调度，Alice 在被调用时回复；这是当前实现方式，不是 Base 在研究层面禁止 worker 主动行为的协议限制。

- 初始只拥有 private relevant context `E_A`。
- 不知道原始问题 `Q`。
- 看不到 Bob 的证据、prompt 或 Bob 与 Celab 的对话。
- v1 调度器在收到 Celab 发给该 worker 的请求后调用其生成一次回复。
- 依据自己的证据以及自己与 Celab 的局部历史作答。
- 回复正文保持自由，可以自然表达追问或澄清需求；v1 不额外实现 worker 自主触发调用的调度机制。
- 不应使用其可见信息无法支持的内容；若无法确定，可以自然地说明无法确定。

### 3.2 Bob

Bob 与 Alice 完全对称。

- 初始只拥有 private relevant context `E_B`。
- 不知道原始问题 `Q`。
- 看不到 Alice 的证据、prompt 或 Alice 与 Celab 的对话。
- v1 调度器在收到 Celab 发给该 worker 的请求后调用其生成一次回复。
- 依据自己的证据以及自己与 Celab 的局部历史作答。
- 回复正文保持自由，可以自然表达追问或澄清需求；v1 不额外实现 worker 自主触发调用的调度机制。
- 不应使用其可见信息无法支持的内容；若无法确定，可以自然地说明无法确定。

### 3.3 Celab

Celab 是问题持有者、协调者和最终答案生成者；v1 调度以 Celab 发起通信为主要流程。

- 初始只拥有原始问题 `Q`。
- 初始看不到 `E_A` 或 `E_B` 的任何内容。
- 能分别向 Alice 或 Bob 发起请求。
- 负责识别问题所需事实，把信息需求写成发给 worker 的请求，汇总回复，并决定何时回答。
- 能看到自己与 Alice、Bob 的两条完整对话历史。
- 只有 Celab 可以正常终止 trajectory，并输出最终答案。
- 不要求必须询问 Alice 和 Bob 各一次；询问顺序、询问对象、询问次数和消息内容都由 Celab 自己决定。

### 3.4 “三个独立 Agent”的实现含义

Alice、Bob、Celab 可以使用同一个 Gemma-3-1B-IT checkpoint，底层也可以共享一份冻结权重以节省显存，但在实验语义和程序抽象中必须是三个独立 `Agent` 对象，至少拥有独立的：

```text
system prompt
visible context
conversation history
generation call
RNG/generation state as applicable
```

不能用一个全局对话上下文轮流切换角色。共享模型权重不等于共享 Agent memory。

## 4. 数据选择与证据分配

### 4.1 固定问题集合

从 HotpotQA validation split 中筛选 100 个问题。筛选完成后固定并保存清单；所有后续 Base 重跑和方法比较都使用同一批 `question_id`。

每条样本至少保存：

```text
question_id
question
gold_answer
HotpotQA metadata
supporting facts / supporting document identifiers
alice_context
bob_context
partition metadata
```

### 4.2 Cross-agent evidence partition

选择适合两跳、跨 Agent 信息交换的样本。对每题的两个必要 relevant contexts，强制分配为：

```text
Alice <- relevant context 1
Bob   <- relevant context 2
Celab <- question only
```

需要满足的实验意图是：

```text
E_A + E_B -> answer
```

并尽量排除以下样本：

```text
E_A alone -> answer
E_B alone -> answer
```

禁止把两个必要 supporting contexts 都交给同一个 worker。不要随机分割段落后假定分割有效；实现中必须检查 supporting-document 分配结果。

由于“单个 context 是否在语义上足以回答”不总能靠字符串规则可靠判断，第一版可组合使用 HotpotQA supporting facts、问题类型规则和人工抽查。筛选规则、筛选 seed、被排除题目及排除原因都应保存，保证样本集合可复现。

### 4.3 不加入 distractors

Alice 和 Bob 各自只收到分给自己的 relevant context，不额外加入 distractor documents。本实验因此聚焦：

```text
communication + coordination + aggregation
```

本阶段不测 local retrieval。

## 5. 信息隔离与可见历史

必须分别维护 `H_A`、`H_B`、`H_C`，不能向三个 Agent 广播全局 trace。

### 5.1 Alice 的模型输入

```text
Alice system prompt
Alice private evidence E_A
only Celab <-> Alice messages so far
current request from Celab
```

严禁包含：原始问题（除非 Celab 在消息正文中选择透露其部分内容）、`E_B`、Bob 的回复、Celab 与 Bob 的历史。

### 5.2 Bob 的模型输入

```text
Bob system prompt
Bob private evidence E_B
only Celab <-> Bob messages so far
current request from Celab
```

严禁包含：原始问题（除非 Celab 在消息正文中选择透露其部分内容）、`E_A`、Alice 的回复、Celab 与 Alice 的历史。

### 5.3 Celab 的模型输入

```text
Celab system prompt
original question Q
all prior Celab <-> Alice messages
all prior Celab <-> Bob messages
```

Celab 只能通过 Alice/Bob 实际发送的回复获得证据信息，不能直接读取 private evidence。

### 5.4 允许的信息传播

Alice/Bob “不知道原始问题”指系统绝不直接把 `Q` 放入其初始 prompt 或上下文。Celab 可以自行在请求正文中透露完成请求所需的问题信息；这部分文字是正常通信，也计入 Celab 的生成 token。

同理，Celab 可以把从一方得到的信息写入发给另一方的后续请求。此时信息是经由 Celab 显式传播，而不是由共享 memory 泄漏。

## 6. 通信拓扑与调度

通信图为 star topology：

```text
Alice <----> Celab <----> Bob

Alice  -X-> Bob
Bob    -X-> Alice
```

v1 为实现简单采用 Celab-driven sequential asynchronous scheduling，一次执行一个 routed request。一次正常循环如下：

1. 调用 Celab；Celab 选择询问 Alice、询问 Bob或给出最终答案。
2. 若 Celab 选择 worker，调度器只调用该 worker 一次。
3. worker 回复后，把回复加入该 worker 的局部历史和 Celab 的历史。
4. 再次调用 Celab，让它根据新信息做下一次决定。

v1 调度器一次执行一个路由请求，并在 worker 回复后继续调用 Celab。这些是当前实现的调度选择，不应提升为 Base 禁止并行通信或 worker 主动行为的研究级协议限制；Base 不对自然语言正文增加额外约束。

## 7. 消息格式、身份与最小路由标记

Base 的原则是 **minimum control, maximum free communication**。路由和终止需要机器可读标记，其余正文完全自由。

### 7.1 Celab 请求 Alice

```text
Celab: <TO>ALICE</TO> message body
```

### 7.2 Celab 请求 Bob

```text
Celab: <TO>BOB</TO> message body
```

### 7.3 Worker 回复

```text
Alice: message body
```

或：

```text
Bob: message body
```

### 7.4 Celab 最终答案

```text
Celab: <FINAL>answer</FINAL>
```

每条消息必须显式表明 speaker identity。按照当前要求，prompt 应要求模型输出自己的 `Alice:`、`Bob:` 或 `Celab:` 前缀；日志层同时必须独立保存结构化 `speaker` 字段，不能只依赖文本前缀判断说话者。

路由 parser 只承担调度所需的最小解析工作：识别 `<TO>ALICE</TO>`、`<TO>BOB</TO>` 或 `<FINAL>...</FINAL>`。不要对消息正文增加词数限制、句式限制、事实格式或策略限制，也不要对中间消息的正确性打分。

无法解析 Celab 输出时，默认不进行自动格式修复：

- 保存未经修改的 raw output，并记录 `parse_status` 和 `parse_error`。
- v1 调度器一次执行一个可唯一识别的动作；无法唯一识别时，不猜测路由目标、不静默改写输出。
- v1 的最小失败处理为结束当前 run，记录 `termination_reason = "parse_error"`、`natural_termination = false`、`final_answer = null`；F1/EM 按官方 evaluator 对空预测处理。该次实际调用的 input/output tokens 和正常 decision step 仍照常计入。
- 解析错误本身不触发重试、格式修复提示或 forced final；记录并持久化失败，不将其静默丢弃。第 9 节的 20-step cap forced final 保持独立，解析失败结束的 run 不再进入该流程。
- 如以后采用其他 retry/failure handling，须另行明确并记录配置与实验版本，不能作为 Base 默认行为隐式加入。

格式错误统计属于实现诊断，不属于对自然语言正文施加协议优化。

## 8. Decision step 的精确定义

一次 **decision step** 定义为：

> Celab 在正常交互阶段被调用一次，并生成下一步动作。

v1 调度层识别以下三种执行结果：

```text
Ask(Alice)
Ask(Bob)
Final(answer)
```

Alice 或 Bob 的回复不增加 decision step。

示例：

```text
Step 1: Celab -> Alice request
        Alice -> Celab response

Step 2: Celab -> Bob request
        Bob -> Celab response

Step 3: Celab -> <FINAL>Canada</FINAL>
```

该 trajectory 的统计为：

```text
decision_steps = 3
num_alice_queries = 1
num_bob_queries = 1
num_total_queries = 2
num_messages = 5
```

正常自然终止时通常有：

```text
decision_steps = num_total_queries + 1
```

但代码必须分别直接计数，不能依赖公式反推，因为异常格式、强制终止和运行错误会破坏该关系。

## 9. 正常终止、上限与强制回答

### 9.1 正常终止

当且仅当 Celab 在正常 decision step 中输出可解析的：

```text
<FINAL>answer</FINAL>
```

trajectory 正常结束，并记录：

```text
termination_reason = "natural_final"
natural_termination = true
cap_reached = false
```

Alice 或 Bob 输出 `<FINAL>` 不得终止整个 trajectory。

### 9.2 20-step safety cap

正常交互最多允许：

```text
max_decision_steps = 20
```

如果正常交互完成第 20 个 Celab decision step，仍未产生合法 final，且未因解析错误等异常结束：

```text
cap_reached = true
natural_termination = false
```

停止进一步询问 Alice/Bob，并执行一次 forced-final operation。

### 9.3 Forced-final operation

向 Celab 添加固定 controller instruction，要求它仅根据当前已有信息给出最佳答案，并使用：

```text
<FINAL>answer</FINAL>
```

这次调用：

- 不计为第 21 个正常 decision step；`decision_steps` 保持 20。
- 单独记录 `forced_final_calls = 1`。
- 其 input/output token 仍计入该 run 的 token 总量，因为它真实发生了推理与生成。
- 保存 raw forced-final output 和解析状态。
- 若仍无法提取 final，令 `final_answer = null`，F1/EM 按官方 evaluator 对空预测处理，并记录 `termination_reason = "forced_final_parse_failure"`。

强制提示词必须固定，不能根据题目或轨迹人工改写。

## 10. 模型与生成配置

三个 Agent 使用相同的基础模型与相同的 decoding 超参数：

```yaml
model: Gemma-3-1B-IT
do_sample: true
temperature: 0.6
top_p: 0.95
max_new_tokens: 2048
max_decision_steps: 20
runs_per_question: 10
```

要求：

- 保留模型原生 EOS；生成遇到 EOS 时正常停止。
- `max_new_tokens=2048` 是单次生成的工程安全上限，不是通信预算，也不是优化约束。
- 不设置更小的 per-message token budget。
- `top_k`、repetition penalty、chat template、dtype、量化方式、模型 revision、tokenizer revision 等未在研究设计中指定的参数，应使用明确的固定值或库默认值，并完整写入运行配置，不能在不同 run 或方法之间变化。
- 保存软件包版本、硬件信息和实际加载的模型标识，以便复现。

### 10.1 Generation cap

每次模型调用都检查是否因为达到 `max_new_tokens` 而停止。不要仅凭输出长度猜测，应优先读取 generation finish reason；本地生成接口没有 finish reason 时，再结合 generated ID 长度与 EOS 状态判断。

至少记录：

```text
generation_cap_reached
generation_cap_agents
generation_cap_events
```

decision-step cap 防止反复对话；generation cap 防止某一次生成失控。两者必须分开统计。

### 10.2 Seeds 与十次独立运行

每题运行 10 次，使用 10 个预先固定的不同 seed。建议用一个全局 seed 清单，例如 `0..9`，并让同一个 `run_index` 在所有问题和后续对比方法中使用相同 seed。

每次 run 保存：

```text
sample_selection_seed
run_seed
run_index
question_id
full resolved config
```

程序应设置相关随机源（如 Python、NumPy、PyTorch CPU/CUDA）。若底层算子仍非确定性，也应记录环境并在报告中说明可复现边界。

## 11. Prompt 原则

Base prompt 必须中性，只定义身份、可见信息、职责、路由和终止语义。不要加入任何会主动改变通信长度或教导策略的内容。

禁止加入：

```text
Be concise.
Use as few tokens as possible.
Communicate efficiently.
Only provide the relevant fact.
Explain thoroughly.
Think step by step.
Ask both agents.
First ask Alice, then ask Bob.
Verify every fact.
```

也不要照抄 OPTIMA 中鼓励极短消息、以 token 数施加压力或建议使用压缩格式的 prompt。

### 11.1 Alice system prompt：语义要求

Alice 的最终逐字 prompt 可以在代码中单独配置，但语义必须只包含：

```text
- You are Alice.
- 下面是只有你能看到的 private evidence。
- 你会收到 Celab 的请求。
- 只依据 private evidence 和 Alice-Celab history 回复。
- 你没有直接看到原始问题。
- 不捏造可见信息无法支持的内容；无法判断时可以说明。
- 输出时明确标识 Alice 身份。
```

### 11.2 Bob system prompt：语义要求

与 Alice 对称，只替换身份和 private evidence。

### 11.3 Celab system prompt：语义要求

Celab 的 prompt 只需说明：

```text
- You are Celab.
- 给出原始问题。
- Alice 与 Bob 各有 Celab 看不到的 private evidence。
- Celab 可自由选择向其中一人请求信息。
- 请求必须带对应 <TO> marker。
- 信息足够时使用 <FINAL>answer</FINAL>。
- 输出时明确标识 Celab 身份。
```

不要规定 Celab 必须问谁、必须问几次、如何分解问题或何时应认为证据充分。

### 11.4 Prompt 版本控制

正式运行前把三份逐字 prompt 作为独立模板文件或配置常量保存，并为其计算内容 hash。每个 run 记录 prompt version/hash。实验中途修改 wording 必须产生新的 experiment version，不能混入同一结果集合。

## 12. Token accounting

### 12.1 主指标：Total Generated Tokens

主效率指标定义为整条 trajectory 中三个 Agent 实际新生成 token 的总和：

```text
#GenTok = alice_generated_tokens
        + bob_generated_tokens
        + celab_generated_tokens
```

其中 Celab 部分包括：

- 正常请求输出；
- 正常 final 输出；
- 无法解析的原始输出（若发生）；
- forced-final 输出（若发生）。

该指标对应当前实验参考的 OPTIMA inference/generated-token 视角。报告中必须写明 `#Tok` 指 generated tokens，避免与 API billing 的 input + output tokens 混淆。

### 12.2 计数方法

必须使用实验实际采用的 Gemma tokenizer，并直接统计模型返回的 generated token IDs：

```text
generated_token_count = len(generated_ids_after_input)
```

不要使用字符数、单词数，也不要将 decoded text 重新 tokenize 后作为主计数。模型实际生成的 speaker prefix、`<TO>` marker、`<FINAL>` marker、正文和 EOS 的计数规则必须在代码中保持一致；建议保存 generated IDs 或足够重建计数的信息，并明确 EOS 是否包含在返回序列及统计中。

### 12.3 Input tokens

每次 inference 同时保存实际 input token 数。对一次调用，input tokens 包含该 Agent 此次真正收到的 system prompt、private context、可见 history、当前消息和 chat-template tokens。

汇总：

```text
alice_input_tokens
bob_input_tokens
celab_input_tokens
total_input_tokens
total_generated_tokens
total_model_tokens = total_input_tokens + total_generated_tokens
```

`total_input_tokens` 和 `total_model_tokens` 是辅助成本指标，不替代 `#GenTok` 主指标。

### 12.4 保留逐消息 token，供后续分析

必须保存每条模型生成消息的 generated token counts，以及对应 raw output、speaker、recipient 和事件类型，使后续可以从原始日志按明确口径重算 `communication_tokens`。当前不把 `communication_tokens` 定义为正式实验指标，不要求计算或保存其聚合值，也不在本阶段规定 final、路由标记等是否属于 communication 的额外统计口径。

主效率指标保持 `total_generated_tokens`，辅助输入成本指标保持 `total_input_tokens`。

框架自动创建的结构化日志字段不计入生成 token；若 speaker prefix 是模型实际生成的，则计入生成 token。

## 13. 最终答案提取与效果评估

### 13.1 Answer extraction

只从 Celab 的最终输出中提取 `<FINAL>` 与 `</FINAL>` 之间的文本：

```text
Celab: <FINAL>Canada</FINAL>
             ^^^^^^
prediction = "Canada"
```

不得把 Celab 的整段输出或中间推理作为预测答案。保存：

```text
final_raw_output
final_answer_extracted
final_parse_status
```

### 13.2 Answer normalization

使用 HotpotQA 官方 answer evaluation 的 normalization 与 token-level scoring逻辑，通常包括：

```text
lowercasing
punctuation removal
article removal
whitespace normalization
```

优先复用官方 evaluator，并通过固定测试样例验证与官方输出一致。

### 13.3 指标

每个 run 计算：

```text
answer_f1   # primary effectiveness metric
answer_em   # auxiliary effectiveness metric
```

Base 不对中间 communication 的事实正确性、简洁性或格式风格评分。Supporting-fact EM/F1 和 joint metrics 不是本阶段主评估项；原始 supporting facts 仍需保存，以便以后做 failure analysis。

## 14. 交互与终止指标

每个 run 至少计算：

```text
decision_steps
num_alice_queries
num_bob_queries
num_total_queries
num_alice_responses
num_bob_responses
num_messages
forced_final_calls
natural_termination
cap_reached
generation_cap_reached
termination_reason
```

数据集级别：

```text
CapRate = runs with cap_reached / all runs
GenerationCapRate = runs with generation_cap_reached / all runs
NaturalTerminationRate = naturally terminated runs / all runs
```

`num_messages` 应按实际模型生成消息计数。固定 controller instruction 不是 Agent 生成消息，但应作为 controller event 记录；forced-final 的 Celab 输出是模型生成消息。

## 15. Logging schema

建议使用 JSONL：每行一个完整 run；另存 message/event 明细也可以，但必须通过 `run_id` 可无损关联。以下字段是最低要求。

### 15.1 Run identity 与配置

```json
{
  "experiment_id": "...",
  "experiment_version": "...",
  "run_id": "...",
  "dataset": "hotpot_qa",
  "dataset_split": "validation",
  "question_id": "...",
  "run_index": 0,
  "sample_selection_seed": 0,
  "run_seed": 0,
  "model_name": "Gemma-3-1B-IT",
  "model_revision": "...",
  "tokenizer_revision": "...",
  "prompt_version": "...",
  "generation_config": {
    "do_sample": true,
    "temperature": 0.6,
    "top_p": 0.95,
    "max_new_tokens": 2048,
    "max_decision_steps": 20
  }
}
```

### 15.2 Sample 与私有信息

```json
{
  "question": "...",
  "gold_answer": "...",
  "dataset_metadata": {},
  "supporting_facts": [],
  "alice_private_context": "...",
  "bob_private_context": "...",
  "partition_metadata": {}
}
```

### 15.3 Message/event trace

每个事件至少保存：

```json
{
  "event_index": 0,
  "event_type": "model_generation",
  "decision_step": 1,
  "speaker": "celab",
  "recipient": "alice",
  "raw_output": "Celab: <TO>ALICE</TO> ...",
  "parsed_action": "ask_alice",
  "parsed_body": "...",
  "parse_status": "ok",
  "parse_error": null,
  "input_tokens": 0,
  "generated_tokens": 0,
  "finish_reason": "eos",
  "generation_cap_reached": false,
  "visible_history_message_ids": [],
  "timestamp_or_sequence": 0
}
```

对 controller event（cap、解析失败终止等）也写入 trace，并明确 `speaker = "controller"`、`generated_tokens = 0`。

### 15.4 Run 结果与聚合字段

```json
{
  "final_raw_output": "...",
  "final_answer": "...",
  "answer_f1": 0.0,
  "answer_em": 0.0,
  "alice_input_tokens": 0,
  "alice_generated_tokens": 0,
  "bob_input_tokens": 0,
  "bob_generated_tokens": 0,
  "celab_input_tokens": 0,
  "celab_generated_tokens": 0,
  "total_input_tokens": 0,
  "total_generated_tokens": 0,
  "decision_steps": 0,
  "num_alice_queries": 0,
  "num_bob_queries": 0,
  "num_total_queries": 0,
  "num_messages": 0,
  "forced_final_calls": 0,
  "natural_termination": true,
  "cap_reached": false,
  "generation_cap_reached": false,
  "generation_cap_agents": [],
  "termination_reason": "natural_final",
  "error": null
}
```

为了以后能够重算指标，不应只保留聚合数字；必须保留 raw outputs、message-level token counts、可见历史引用、parser 结果和完整配置。

## 16. 汇总与报告

保留全部 1000 个 run 的原始值。至少报告：

- Answer F1：mean、standard deviation。
- Answer EM：mean、standard deviation。
- Total Generated Tokens：mean、standard deviation，以及分 Agent 均值。
- Total Input Tokens：mean、standard deviation。
- Decision Steps：mean、standard deviation。
- Alice/Bob/total query counts：mean、standard deviation。
- Cap Rate、Generation Cap Rate、Natural Termination Rate。

建议同时生成 per-question 汇总：先对同一题的 10 次 run 计算均值和标准差，再提供 dataset-level 汇总。原始 run-level 分布必须保留，不能只保存平均值。

未来比较新方法时，应在相同问题、相同 run-index seed 和相同基础模型配置上做 paired comparison。主要结果应分别展示正确性和 token，而不是把两者强行合成单一 efficiency score。

## 17. 实现状态机参考

```text
initialize Alice(E_A), Bob(E_B), Celab(Q)
decision_steps = 0

while decision_steps < 20:
    call Celab
    decision_steps += 1
    log input/output tokens and raw output

    if action == FINAL:
        extract answer
        terminate naturally

    if action == ASK_ALICE:
        call Alice once
        append only to H_A and H_C
        log response
        continue

    if action == ASK_BOB:
        call Bob once
        append only to H_B and H_C
        log response
        continue

    if action cannot be parsed:
        preserve raw output and log parse_status / parse_error
        do not guess, rewrite, retry, or call forced final
        termination_reason = "parse_error"
        natural_termination = false
        final_answer = null
        terminate interaction loop

if step 20 completed without natural final and run has not ended on error:
    cap_reached = true
    call Celab once with fixed forced-final instruction
    do not increment normal decision_steps
    count all forced-final model tokens
    extract answer if possible

evaluate final answer with HotpotQA F1/EM (null -> empty prediction)
persist complete run record
```

## 18. 必须验证的实现性质

至少为以下性质编写有意义的自动测试或断言：

1. Alice 的任意模型输入不包含 Bob private context 或 Bob-Celab messages。
2. Bob 的任意模型输入不包含 Alice private context 或 Alice-Celab messages。
3. Alice/Bob 初始输入不直接包含原始问题字段。
4. Celab 初始输入不包含任何 private evidence。
5. Celab 可以在自己的消息中主动转述信息，而这种转述不被误报为 memory leakage。
6. Alice/Bob 无法直接通信；v1 由 Celab 驱动调度，每次执行一个路由请求，worker 被调用后回复。这不构成禁止 worker 主动行为或并行通信的研究级协议限制。
7. worker reply 不增加 `decision_steps`。
8. 自然 final 所在的 Celab 调用计入 `decision_steps`。
9. 20 步后 forced-final 不把 `decision_steps` 增加到 21，但其 token 被计入总量。
10. 达到 2048 token 的 generation event 与 decision cap 分开记录。
11. `total_generated_tokens` 等于所有模型 generation event 的 generated-token 总和。
12. evaluator 只使用 `<FINAL>` 内容，并与 HotpotQA 官方 normalization/F1 实现一致。
13. 相同 question、seed、配置在可确定的运行环境下能够复现；所有配置差异可从日志识别。
14. 100 个样本中，每题的两个必要 supporting contexts 被分给不同 worker。
15. 无法解析 Celab 输出时保留 raw output 和 parse error，不自动修复、重试或 forced final；失败 run 与其已产生的 token 和 decision steps 完整保存。

## 19. 当前明确暂不实现的内容

以下内容不属于本 Base 的第一阶段：

- 2WikiMultiHopQA、TriviaQA、CBT 或其他数据集。
- distractor documents 与 local retrieval。
- C-only baseline；它以后可用于衡量 parametric knowledge，但不阻止 Base 先运行。
- single-agent full-evidence baseline。
- oracle communication baseline。
- Alice 与 Bob 直接通信。
- 强制 Celab 询问两名 worker。
- communication token budget、短回复上限或 adaptive budget。
- concise prompt、chain-of-thought prompt、详细解释提示。
- 复杂 routing policy、证据压缩、learned protocol。
- SFT、DPO、RL 或其他训练。
- 对中间消息做正确性评分。
- supporting-fact/joint metric 作为当前主结果。

这些项目只能作为后续独立扩展或 ablation，不应悄悄加入 Base 配置。

v1 暂未实现 worker 自主触发调用的调度机制或并行请求执行；这是实现范围说明，不是 Base 的理论禁止项。worker 在回复正文中自然提出追问或澄清需求不受额外限制。自动格式修复重试默认不启用，解析失败按第 7 节的显式最小处理执行。

## 20. 完成标准

实现完成需满足：

- 能从固定 HotpotQA 样本清单加载一题并正确构造三份隔离上下文。
- 能按 star topology 跑完一个自然终止 trajectory。
- 能正确解析两个路由标记和 final 标记，同时保留自由消息正文。
- 能准确统计 decision steps、query counts、每次 input/output tokens 和三名 Agent 的生成 token。
- 能触发并正确记录 20-step forced final 与 2048-token generation cap。
- 能输出可重算的完整 JSONL trace。
- 能使用 HotpotQA 标准逻辑计算 Answer F1 和 EM。
- 能以固定样本与 10 个 seeds 执行 100 × 10 = 1000 条 trajectories，并生成汇总报告。

满足以上条件后，即得到本研究第一阶段的 HotpotQA 三智能体自由通信 Base。
