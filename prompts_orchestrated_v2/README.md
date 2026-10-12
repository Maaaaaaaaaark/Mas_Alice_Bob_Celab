# 协调者协议 v2:提示词与运行规则

模型:C = gemma-3-4b-it,A/B = gemma-3-1b-it。所有提示词用英文,占位符为 `$name`(与现有 `PromptSet` 一致)。
Gemma 的聊天模板没有 system 角色,所以每个提示词整体作为**一条 user 消息**发送。

只换提示词不够:下面的"框架规则"必须同时实现,否则上次的问题(问错人、编造内容到达 C)还会出现。

## 1. 流程

```
1. 分配   C(c_allocate)                      -> 校验 -> A 的文档, B 的文档
2. 循环   asks_left = 3, log = []
          如果 asks_left == 0: C(c_turn_final),结束
          out = C(c_turn)
          ANSWER  -> 结束
          ASK A / ASK B / ASK BOTH:
              对每个被问的 worker:
                  reply = W(worker_answer, 它自己的文档, C 的问题)
                  reply = verify_reply(reply, 它自己的文档)   # 只保留核对通过的引用
                  log 追加两行:问题、回答
              asks_left -= 被问的 worker 数(ASK BOTH 算 2)
```

## 2. 占位符格式

| 占位符 | 格式示例 |
|---|---|
| `$titles` | 每行一个:`D1: Left-wing uprisings against the Bolsheviks` |
| `$titles_a` / `$titles_b` | 一行:`D2 Louise Bryant; D4 Alexander Kerensky; ...` |
| `$documents` | 每篇:`D5: Title: Russian Civil War` 换行后接正文,篇与篇之间空一行 |
| `$log` | 每条消息一行,例如 `C to B: When did the Russian Civil War end?` / `B to C: QUOTE [D5] (verified): "..."`;为空时写 `(nothing yet)` |
| `$asks_left` | 剩余可问次数(整数) |
| `$name` | `A` 或 `B` |
| `$c_question` | C 问这个 worker 的那一句 |

worker 的输入**不包含原问题**,只包含 C 的问题。这样 1B 不会试图回答整道题。可以作为开关,以后对比。

## 3. 解码设置(全部贪心)

| 调用 | max_new_tokens | 预填 | 停止 | 解析 |
|---|---|---|---|---|
| c_allocate | 40 | `A:` | 第二个换行 | 取 `A:`、`B:` 两行的 D 编号 |
| c_turn | 60 | 无 | 匹配到第一条有效行后停 | 第一行以 `ASK A:` / `ASK B:` / `ASK BOTH:` / `ANSWER:` 开头的 |
| c_turn_final | 16 | `ANSWER:` | 换行 | 预填之后的文本 |
| worker_answer | 100 | 无 | 连续两个换行 | 交给 `verify_reply` |
| baseline_single_reader | 16 | `Answer:` | 换行 | 预填之后的文本 |
| baseline_title_select | 16 | `READ:` | 换行 | 最多 2 个 D 编号 |

预填的做法:`apply_chat_template(..., add_generation_prompt=True)` 得到的 token 后面,再接上 `tokenizer.encode(预填, add_special_tokens=False)`。

## 4. 校验与兜底(每一次都要记录)

- **分配**:A、B 各 5 篇,不重复、不遗漏。不合法时,保留 A 行里前 5 个合法且不重复的编号,其余给 B,并记一次"分配不合法"。
- **c_turn 解析失败**:记一次,直接进入 c_turn_final。
- **剩 1 次时却用了 ASK BOTH**:照常发给两个人,然后 asks_left 记为 0,记一次。
- **追问时,worker 的输入必须带上它自己的文档**。上次 A 的回答里没有一句来自文档,先确认这一点。

## 5. 引用核对(最关键)

worker 回答里的每一条引用,都要能在**这个 worker 自己的**文档里找到,才交给 C;找不到就丢掉。全部丢掉时,交给 C 的是 `NOT FOUND`。
引用标错了文档编号、但确实出自它的另一篇文档时,接受并改正编号。

```python
import difflib
import re

QUOTE_RE = re.compile(r'^QUOTE\s*\[(D\d+)\]\s*:\s*(.+)$')


def _words(s: str) -> list:
    s = s.lower().replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r"[^\w\s]", " ", s)
    return s.split()


def _best_window_ratio(q: list, t: list) -> float:
    """quote 和原文中等长窗口的最高相似度(词级),避免长原文里零散匹配造成误判。"""
    best = 0.0
    for size in (len(q) - 1, len(q), len(q) + 1):
        if size <= 0:
            continue
        for i in range(max(1, len(t) - size + 1)):
            best = max(best, difflib.SequenceMatcher(None, q, t[i:i + size]).ratio())
    return best


def verify_reply(reply: str, docs: dict, min_ratio: float = 0.9, min_words: int = 4):
    """docs: {"D5": "Title: Russian Civil War\\n正文..."}。
    返回 (交给 C 的文本, 统计)。只保留核对通过的引用,最多 2 条。"""
    kept, n_claimed = [], 0
    for line in reply.splitlines():
        m = QUOTE_RE.match(line.strip())
        if not m:
            continue
        n_claimed += 1
        raw = m.group(2).strip().strip('"').strip()
        q = _words(raw)
        if len(q) < min_words:
            continue
        cited = m.group(1)
        order = [cited] + [d for d in docs if d != cited]   # 先查标注的那篇
        for doc_id in order:
            if doc_id in docs and _best_window_ratio(q, _words(docs[doc_id])) >= min_ratio:
                kept.append(f'QUOTE [{doc_id}] (verified): "{raw}"')
                break
    stats = {"claimed": n_claimed, "verified": len(kept),
             "said_not_found": reply.strip().upper().startswith("NOT FOUND")}
    return ("\n".join(kept[:2]) if kept else "NOT FOUND"), stats
```

## 6. 要跑的四组(同一批 100 道验证题)

| 组 | 做法 |
|---|---|
| 交互(本协议) | 分配 → 最多问 3 次 → 作答 |
| 一次性 | 分配 → 把**原问题**同时发给 A、B(`$c_question` = 原问题)→ c_turn_final |
| 4B 读全部 | baseline_single_reader,10 篇全给 |
| 4B 看标题挑读 | baseline_title_select → 选中的文档交给 baseline_single_reader |

## 7. 每道题要记录的

- EM、F1
- 每次调用:角色、模型、输入 token、输出 token;汇总为 4B token、1B token、加权成本 = 1×(1B token) + 4×(4B token)
- 分配是否合法;两篇 supporting 文档是否被分到同一个 worker(只用于分析)
- 每题追问次数;每次追问是否问到了手里有 supporting 文档的 worker(只用于分析)
- worker 每次回答的 `claimed`、`verified`、`said_not_found`;**编造率** = 未通过核对的引用 / 声称的引用;`claimed == 0` 且 `said_not_found` 为假,说明 worker 没按格式回答(单独统计**格式违规率**)
- 完整对话原文(至少保存前 20 道题)
