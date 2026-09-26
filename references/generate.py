"""Deterministic multi-agent trajectory generation for the Qwen OPTIMA pipeline.

Replaces ``train/datagenerate.py::vllm_data_generate*`` for the new run-config
pipeline:

- per-trajectory derived RNGs -> reproducible sampling (vLLM per-request seed)
- model-agnostic :class:`VllmAgent` (no hardcoded chat template)
- speaker-aware memory: [system, own turns=assistant, partner turns=user]
- per-turn metadata (:class:`Turn`) recorded for transcripts and auditing
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import random
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent.agent import VllmAgent
from answerParser.parser import hotpot_qa_parser, math_parser
from message.message import llmMessage, Turn
from utils.prompt_template import (
    prompt,
    prompt_multi_arc_first,
    prompt_multi_arc_second,
    get_prompt_pool,
)
from utils.run_config import RunConfig

_lock = threading.Lock()

# iteration-0 name-prefix instruction appended to debate prompts
NAME_SUFFIX = '\n 3. You must begin your response with "${name}:".'


def _remove_name_prefix_instruction(text: str) -> str:
    """Remove display-only speaker-prefix instructions for the no-prefix
    ablation. Speaker identity remains available in Turn.speaker and chat
    roles, so routing and transcript attribution are unaffected."""
    patterns = (
        r'^\s*\d+\.\s*You must begin your response with ["\\]*\$\{name\}:["\\]*\.\s*$',
        r'^\s*\d+\.\s*You should start your utterance with ["\\]*\$\{name\}:["\\]*\.\s*$',
    )
    lines = text.splitlines()
    return "\n".join(
        line for line in lines
        if not any(re.match(pattern, line) for pattern in patterns)
    )


@dataclass
class TaskSample:
    task_id: int
    question: str
    answer: Any  # str, or list[str] for multi-answer (HotpotQA)
    context_first: List[str]
    context_second: List[str]
    data_type: str = "qa"          # "qa" (hotpot_qa) | "debate" (arc) | "math"
    dataset_name: str = "hotpot_qa"


@dataclass
class Trajectory:
    task_id: int
    trajectory_id: int
    question: str
    answer: Any
    context_first: List[str]
    context_second: List[str]
    data_type: str
    dataset_name: str
    score_type: str
    system_first: str = ""
    system_second: str = ""
    turns: List[Turn] = field(default_factory=list)
    conversation: List[str] = field(default_factory=list)  # turn contents (+ "large" sentinel)
    final_answer: str = ""
    token_count: int = 0
    termination_reason: str = "max_round"
    # reward / selection fields, filled by reward/scorer.py + dataset_build.py
    reward: Optional[float] = None
    correct_score: Optional[float] = None
    token_score: Optional[float] = None
    ppl_score: Optional[float] = None
    name_penalty: Optional[float] = None
    selected: bool = False
    rank: Optional[int] = None

    def to_result(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "trajectory_id": self.trajectory_id,
            "question": self.question,
            "answer": self.answer,
            "context_first": self.context_first,
            "context_second": self.context_second,
            "data_type": self.data_type,
            "dataset_name": self.dataset_name,
            "score_type": self.score_type,
            "system_first": self.system_first,
            "system_second": self.system_second,
            "turns": [t.dict() for t in self.turns],
            "conversation": list(self.conversation),
            "final_answer": self.final_answer,
            "token_count": self.token_count,
            "termination_reason": self.termination_reason,
            "reward": self.reward,
            "correct_score": self.correct_score,
            "token_score": self.token_score,
            "ppl_score": self.ppl_score,
            "name_penalty": self.name_penalty,
            "selected": self.selected,
            "rank": self.rank,
        }

    @classmethod
    def from_result(cls, result: Dict[str, Any]) -> "Trajectory":
        turns = [
            Turn(**t) for t in result.get("turns", [])
        ]
        return cls(
            task_id=result["task_id"],
            trajectory_id=result.get("trajectory_id", 0),
            question=result["question"],
            answer=result["answer"],
            context_first=result.get("context_first", []),
            context_second=result.get("context_second", []),
            data_type=result.get("data_type", "qa"),
            dataset_name=result.get("dataset_name", "hotpot_qa"),
            score_type=result.get("score_type", "f1-score"),
            system_first=result.get("system_first", ""),
            system_second=result.get("system_second", ""),
            turns=turns,
            conversation=result.get("conversation", []),
            final_answer=result.get("final_answer", ""),
            token_count=result.get("token_count", 0),
            termination_reason=result.get("termination_reason", "max_round"),
            reward=result.get("reward"),
            correct_score=result.get("correct_score"),
            token_score=result.get("token_score"),
            ppl_score=result.get("ppl_score"),
            name_penalty=result.get("name_penalty"),
            selected=result.get("selected", False),
            rank=result.get("rank"),
        )


def task_rng(seed: int, iteration: int, task_id: int) -> random.Random:
    return random.Random(f"{seed}:{iteration}:{task_id}")


def traj_rng(seed: int, iteration: int, task_id: int, traj_id: int) -> random.Random:
    return random.Random(f"{seed}:{iteration}:{task_id}:{traj_id}")


def _has_iteration0_marker(text: str) -> bool:
    return (
        "You should start your utterance with" in text
        or "You must begin your response with" in text
    )


def _prepare_prompts(
    cfg: RunConfig, data_type: str, iteration: int, rng: random.Random
) -> Tuple[str, str]:
    """Return (first_template, second_template). Mirrors the old vllm path:
    - qa: shared base template; optional prompt pool
    - debate: separate first/second templates; iteration 0 appends the
      name-prefix instruction and (if a pool is configured) diversifies the
      first agent's template.
    """
    if data_type == "debate":
        first_prompt = prompt_multi_arc_first
        second_prompt = prompt_multi_arc_second
        if iteration == 0:
            second_prompt = second_prompt + NAME_SUFFIX
            if not _has_iteration0_marker(first_prompt):
                first_prompt = first_prompt + NAME_SUFFIX
        pool: List[str] = []
        if iteration == 0 and cfg.prompt_pool_path:
            pool = get_prompt_pool(cfg.prompt_pool_path)
        if pool:
            first_prompt = rng.choice(pool)
        if not cfg.require_name_prefix:
            first_prompt = _remove_name_prefix_instruction(first_prompt)
            second_prompt = _remove_name_prefix_instruction(second_prompt)
        return first_prompt, second_prompt

    template = prompt
    if cfg.prompt_pool_path and iteration == 0:
        pool = get_prompt_pool(cfg.prompt_pool_path)
        if pool:
            template = rng.choice(pool)
    if not cfg.require_name_prefix:
        template = _remove_name_prefix_instruction(template)
    return template, template


def _parse_answer(turn: Turn, data_type: str) -> Optional[str]:
    if data_type == "math":
        return math_parser(turn)
    return hotpot_qa_parser(turn)


def generate_trajectory(
    cfg: RunConfig, task: TaskSample, traj_id: int, iteration: int
) -> Trajectory:
    rng = traj_rng(cfg.seed, iteration, task.task_id, traj_id)
    temperature = cfg.temperature_iter0 if iteration == 0 else cfg.temperature
    seed_provider = lambda: rng.randint(0, 2**31 - 1)

    first_template, second_template = _prepare_prompts(
        cfg, task.data_type, iteration, rng
    )
    score_type = "exact-match" if task.data_type in ("debate", "math") else "f1-score"

    agent_first = VllmAgent(
        url=cfg.alice.url,
        my_model_name=cfg.alice.served_model_name,
        name="Alice",
        temperature=temperature,
        max_tokens=cfg.max_tokens_per_turn,
        seed_provider=seed_provider,
        use_name_prefix=cfg.require_name_prefix,
        merge_system_into_user=cfg.merge_system_into_user,
    )
    agent_second = VllmAgent(
        url=cfg.bob.url,
        my_model_name=cfg.bob.served_model_name,
        name="Bob",
        temperature=temperature,
        max_tokens=cfg.max_tokens_per_turn,
        seed_provider=seed_provider,
        use_name_prefix=cfg.require_name_prefix,
        merge_system_into_user=cfg.merge_system_into_user,
    )
    agent_first.init_system_prompt(
        first_template,
        {
            "name": agent_first.name,
            "partner": agent_second.name,
            "question": task.question,
            "information": "\n".join(task.context_first),
        },
    )
    agent_second.init_system_prompt(
        second_template,
        {
            "name": agent_second.name,
            "partner": agent_first.name,
            "question": task.question,
            "information": "\n".join(task.context_second),
        },
    )

    traj = Trajectory(
        task_id=task.task_id,
        trajectory_id=traj_id,
        question=task.question,
        answer=task.answer,
        context_first=task.context_first,
        context_second=task.context_second,
        data_type=task.data_type,
        dataset_name=task.dataset_name,
        score_type=score_type,
        system_first=agent_first.system_prompt.content,
        system_second=agent_second.system_prompt.content,
    )

    agent_list = [agent_first, agent_second]
    current = 0
    now_round = 0
    final_answer = ""
    final_answer_speaker: Optional[str] = None
    while now_round < cfg.max_round:
        agent = agent_list[current]
        turn = agent.step()
        traj.turns.append(turn)
        traj.conversation.append(turn.content)

        if turn.content == "error":
            traj.termination_reason = "error"
            break
        if turn.token_count >= cfg.max_tokens_per_turn:
            traj.conversation.append("large")
            traj.termination_reason = "large"
            break
        # debate/math turns must carry the speaker prefix (iteration 0 relies
        # on the prompt instruction; enforce it like the old conversation())
        if task.data_type != "qa" and not turn.content.strip().startswith(turn.speaker):
            turn.content = f"{turn.speaker}:{turn.content}"
            traj.conversation[-1] = turn.content
        turn.parsed_answer = _parse_answer(turn, task.data_type)
        if turn.parsed_answer not in (None, ""):
            # Agreement requires confirmation by the *other* agent.  Without
            # the speaker check, Alice repeating her own answer after Bob says
            # only "Correct"/"Agreed" is incorrectly counted as consensus.
            if (
                turn.parsed_answer == final_answer
                and final_answer_speaker is not None
                and turn.speaker != final_answer_speaker
            ):
                traj.termination_reason = "agreement"
                break
            final_answer = turn.parsed_answer
            final_answer_speaker = turn.speaker

        current = (current + 1) % 2
        # partner stores this turn as a "user" message (speaker-aware memory)
        agent_list[current].add_memory(llmMessage(role="user", content=turn.content))
        now_round += 1

    traj.final_answer = final_answer
    traj.token_count = sum(t.token_count for t in traj.turns)
    return traj


def rewrite_sorted_jsonl(path: str) -> None:
    """Atomically rewrite a jsonl sorted by (task_id, trajectory_id)."""
    if not os.path.exists(path):
        return
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(data)
    rows.sort(key=lambda d: (d.get("task_id", -1), d.get("trajectory_id", -1)))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def generate_one_task(
    cfg: RunConfig,
    task: TaskSample,
    iteration: int,
    output_path: str,
) -> None:
    results = []
    for traj_id in range(cfg.explore_count):
        try:
            traj = generate_trajectory(cfg, task, traj_id, iteration)
        except Exception:
            traceback.print_exc()
            traj = Trajectory(
                task_id=task.task_id,
                trajectory_id=traj_id,
                question=task.question,
                answer=task.answer,
                context_first=task.context_first,
                context_second=task.context_second,
                data_type=task.data_type,
                dataset_name=task.dataset_name,
                score_type="exact-match" if task.data_type in ("debate", "math") else "f1-score",
                termination_reason="error",
            )
            traj.conversation.append("error")
        results.append(traj.to_result())
    with _lock:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps({"task_id": task.task_id, "results": results}, ensure_ascii=False)
                + "\n"
            )
    print(f"[generate] task {task.task_id} done ({len(results)} trajectories)")


def generate_all(
    cfg: RunConfig,
    dataloader,
    iteration: int,
) -> int:
    """Generate cfg.explore_count trajectories for each of cfg.sample_count
    tasks. Resumes from tasks already present in the raw output file and
    rewrites it sorted at the end."""
    output_path = cfg.raw_path(iteration)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    done_tasks = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_tasks.add(json.loads(line)["task_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"[generate] iteration {iteration}: {len(done_tasks)} tasks already done")

    tasks: List[TaskSample] = []
    for i in range(cfg.sample_count):
        rng = task_rng(cfg.seed, iteration, i)
        question, answer, context1, context2 = dataloader.sample_once(rng=rng)
        # ``sample_once`` advances sequential dataset loaders such as
        # HotpotQA.  Advance it for completed task ids as well, otherwise a
        # resumed/expanded run would assign dataset row 0 to the first new
        # task id and silently duplicate earlier questions.
        if i in done_tasks:
            continue
        tasks.append(
            TaskSample(
                task_id=i,
                question=question,
                answer=answer,
                context_first=context1,
                context_second=context2,
                data_type=dataloader.data_type,
                dataset_name=dataloader.dataset_name,
            )
        )

    print(f"[generate] {len(tasks)} tasks to run")
    with ThreadPoolExecutor(max_workers=cfg.thread_count) as executor:
        futures = [
            executor.submit(generate_one_task, cfg, task, iteration, output_path)
            for task in tasks
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                traceback.print_exc()

    rewrite_sorted_jsonl(output_path)
    return cfg.sample_count


def write_transcripts(cfg: RunConfig, iteration: int) -> int:
    """Write one readable transcript file per trajectory (task_XXXXX_traj_XX.txt)
    plus a merged all_transcripts.txt, for manual inspection."""
    out_dir = cfg.transcripts_dir(iteration)
    os.makedirs(out_dir, exist_ok=True)

    def _load(path: str) -> Dict[Tuple[int, int], Dict]:
        rows: Dict[Tuple[int, int], Dict] = {}
        if not path or not os.path.exists(path):
            return rows
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for res in data.get("results", []):
                    rows[(data["task_id"], res.get("trajectory_id", 0))] = res
        return rows

    raw = _load(cfg.raw_path(iteration))
    rewarded = _load(cfg.rewarded_path(iteration))
    cleaned = _load(cfg.cleaned_path(iteration))

    merged_parts = []
    count = 0
    for (task_id, traj_id), res in sorted(raw.items()):
        info = dict(res)
        info.update(rewarded.get((task_id, traj_id), {}))
        info.update(cleaned.get((task_id, traj_id), {}))
        lines = [
            f"Task {task_id} / Trajectory {traj_id}",
            f"dataset: {info.get('dataset_name', '')} ({info.get('data_type', '')}) | "
            f"score_type: {info.get('score_type', '')} | termination: {info.get('termination_reason', '')}",
            f"golden answer: {info.get('answer')}",
            f"final answer: {info.get('final_answer', '')}",
            f"token_count: {info.get('token_count', '')}",
        ]
        if info.get("reward") is not None:
            lines.append(
                f"reward: {info['reward']:.4f} = correct {info.get('correct_score')} "
                f"+ token {info.get('token_score')} + ppl {info.get('ppl_score')} "
                f"(name_penalty {info.get('name_penalty')}) | "
                f"selected: {info.get('selected', False)} rank: {info.get('rank')}"
            )
        lines.append("\n---- Alice system prompt ----\n" + info.get("system_first", ""))
        lines.append("\n---- Bob system prompt ----\n" + info.get("system_second", ""))
        for idx, turn in enumerate(info.get("turns", [])):
            lines.append(
                f"\n=== Turn {idx + 1}: {turn.get('speaker')} "
                f"({turn.get('token_count')} tokens, parsed={turn.get('parsed_answer')!r}, "
                f"finish={turn.get('finish_reason')}) ===\n{turn.get('content', '')}"
            )
        if "large" in info.get("conversation", []):
            lines.append("\n[turn exceeded token limit -> 'large']")
        text = "\n".join(lines) + "\n" + "=" * 80 + "\n"
        fname = f"task_{task_id:05d}_traj_{traj_id:02d}.txt"
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            f.write(text)
        merged_parts.append(f"[{fname}]\n{text}")
        count += 1

    with open(os.path.join(out_dir, "all_transcripts.txt"), "w", encoding="utf-8") as f:
        f.write("".join(merged_parts))
    return count
