"""In-process reward scoring for the Qwen OPTIMA pipeline (no Ray, no deploys).

Replicates the scoring rules of the old ``reward/deploy_reward.py`` path
(``get_score_deploy``) with one frozen configured base model held locally:

    R = R_task - lambda_token * (tokens / max_tokens_in_task) + lambda_loss / max_per_turn_loss

Special rules carried over from the old code:
- token_count <= 40          -> token_score = -1
- "large" in conversation    -> token_score = -1
- token_count > 2000 or empty conversation -> ppl_score = -1
- duplicated utterance (after optional name-prefix normalization) -> ppl_score = 0
- when name prefixes are required, mixing/repeating names -> correct_score = 0
"""
from typing import Any, Dict, List, Optional, Tuple
import gc
import json
import os
import re

import torch
import torch.nn as nn

from reward.reward import cal_f1_score
from answerParser.parser import is_equiv
from utils.run_config import RunConfig


def frame_utterance_for_loss(tokenizer, text: str, merge_system_into_user: bool) -> str:
    """Render one utterance as the model's own turn for LM-loss scoring.

    Default framing is a lone assistant turn (Qwen: ``<|im_start|>assistant``).
    Gemma's chat template forbids an assistant-only conversation, so there the
    utterance is framed as an (empty user turn +) assistant turn, which renders
    the equivalent ``<start_of_turn>model`` header. The loss formula itself is
    unchanged in both cases.
    """
    if merge_system_into_user:
        messages = [{"role": "user", "content": ""}, {"role": "assistant", "content": text}]
    else:
        messages = [{"role": "assistant", "content": text}]
    return tokenizer.apply_chat_template(messages, tokenize=False)


class LossScorer:
    """Frozen base model used to compute per-turn LM losses (R_loss term)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        batch_size: int = 16,
        merge_system_into_user: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.batch_size = batch_size
        self.merge_system_into_user = merge_system_into_user
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Gemma tokenizers have no pad token; fall back to eos so padding
        # and the ignore_index masking keep working (Qwen already has one).
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_dtype: Any = "auto"
        if (
            str(device).startswith("cuda")
            and torch.cuda.is_available()
            and not torch.cuda.is_bf16_supported()
        ):
            # Gemma 2 checkpoints default to bf16, which a Tesla T4 cannot
            # execute natively.  This changes storage dtype only; the frozen
            # R_loss formula and all reward rules remain identical.
            model_dtype = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=model_dtype
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def per_turn_losses(self, texts: List[str]) -> List[float]:
        """Mean cross-entropy per text, templated as a single assistant turn
        (the same framing the old pipeline used for its ppl term)."""
        pad_id = self.tokenizer.pad_token_id
        losses: List[float] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            templated = [
                frame_utterance_for_loss(self.tokenizer, t, self.merge_system_into_user)
                for t in chunk
            ]
            inputs = self.tokenizer(
                templated, return_tensors="pt", padding=True, truncation=True
            )["input_ids"].to(self.device)
            with torch.inference_mode():
                logits = self.model(input_ids=inputs).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_id)
            loss = loss_fct(
                # Keep the original cross-entropy reward formula, but compute
                # it in fp32.  Gemma 2 is loaded in fp16 on T4 GPUs and a
                # vocabulary-wide fp16 CE can overflow to inf, incorrectly
                # collapsing lambda2 / max_loss to zero.
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss = loss.view(shift_labels.size())
            per = loss.sum(dim=1) / (shift_labels != pad_id).sum(dim=1).clamp(min=1)
            losses.extend(per.cpu().tolist())
        return losses


def _correct_score(result: Dict[str, Any], tokenizer, score_type: str) -> float:
    answer = result["answer"]
    final_answer = result.get("final_answer", "") or ""
    try:
        if isinstance(answer, list):
            if score_type == "f1-score":
                return max(cal_f1_score(a, final_answer, tokenizer) for a in answer)
            # exact-match: golden answers is a list, model answer is a string
            gold = [a.strip().lower() for a in answer]
            return 1.0 if final_answer.strip().lower() in gold else 0.0
        if score_type == "f1-score":
            return cal_f1_score(answer, final_answer, tokenizer)
        if score_type == "math":
            try:
                return float(
                    is_equiv(answer.strip().lower(), final_answer.strip().lower())
                )
            except Exception:
                return 1.0 if answer.strip().lower() == final_answer.strip().lower() else 0.0
        # exact-match
        return (
            1.0
            if answer.strip().lower() == final_answer.strip().strip("\\").lower()
            else 0.0
        )
    except Exception:
        return 0.0


def _utterance_penalties(
    conversation: List[str],
    result: Dict[str, Any],
    check_name_format: bool = True,
) -> bool:
    """Broken-format checks from the old get_score_deploy. Returns True when
    the utterance list is sane (no name-mix / no duplication)."""
    record: List[str] = []
    for sentence in conversation:
        try:
            text = str(sentence)
        except Exception:
            continue
        if check_name_format:
            if re.search(r"Alice:", text) and re.search(r"Bob:", text):
                result["correct_score"] = 0.0
                return False
            if len(re.findall(r"Alice:", text)) >= 2 or len(re.findall(r"Bob:", text)) >= 2:
                result["correct_score"] = 0.0
                return False
            normalized = re.sub(r"^\s*(Alice|Bob)\s*:\s*", "", text).strip().lower()
        else:
            normalized = text.strip().lower()
        if normalized in record:
            result["ppl_score"] = 0.0
            return False
        record.append(normalized)
    return True


def score_task(
    result: Dict[str, Any],
    loss_scorer: LossScorer,
    max_token_count: int,
    lambda1: float,
    lambda2: float,
    score_type: str = "f1-score",
    cal_ppl: bool = True,
    check_name_format: bool = True,
) -> Dict[str, Any]:
    tokenizer = loss_scorer.tokenizer
    conversation = [c for c in result.get("conversation", []) if c != "large"]
    result["correct_score"] = _correct_score(result, tokenizer, score_type)
    result["token_score"] = 0.0
    result["ppl_score"] = 0.0

    try:
        result["token_score"] = lambda1 * result["token_count"] / max_token_count
    except (KeyError, ZeroDivisionError):
        pass
    if "large" in result.get("conversation", []):
        result["token_score"] = -1.0
    if result.get("token_count", 0) <= 40:
        result["token_score"] = -1.0

    if cal_ppl:
        if result.get("token_count", 0) > 2000 or len(conversation) == 0:
            result["ppl_score"] = -1.0
        elif _utterance_penalties(
            conversation, result, check_name_format=check_name_format
        ) and conversation:
            losses = loss_scorer.per_turn_losses(conversation)
            max_loss = max(losses) if losses else 1.0
            result["ppl_score"] = lambda2 / max_loss
        # duplicated/broken conversations already got ppl_score = 0

    result["reward"] = (
        result["correct_score"] + result["token_score"] + result["ppl_score"]
    )
    return result


def score_all(cfg: RunConfig, iteration: int) -> int:
    """Score every trajectory in the raw file and write the rewarded jsonl."""
    raw_path = cfg.raw_path(iteration)
    rewarded_path = cfg.rewarded_path(iteration)
    os.makedirs(os.path.dirname(rewarded_path), exist_ok=True)

    print(f"[reward] loading frozen reward model from {cfg.reward_model_path}")
    scorer = LossScorer(
        cfg.reward_model_path,
        device=cfg.scorer_device,
        batch_size=cfg.scorer_batch_size,
        merge_system_into_user=cfg.merge_system_into_user,
    )

    tasks: List[Dict[str, Any]] = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"[reward] scoring {len(tasks)} tasks x {cfg.explore_count} trajectories")

    rows: List[Dict[str, Any]] = []
    for task in tasks:
        results = task["results"]
        max_token_count = max(
            [
                r["token_count"]
                for r in results
                if "large" not in r.get("conversation", [])
            ]
            or [1]
        )
        for result in results:
            score_type = result.get("score_type", "f1-score")
            score_task(
                result,
                scorer,
                max_token_count,
                cfg.lambda1,
                cfg.lambda2,
                score_type=score_type,
                cal_ppl=cfg.cal_ppl,
                check_name_format=cfg.require_name_prefix,
            )
        rows.append(task)

    rows.sort(key=lambda d: d.get("task_id", -1))
    with open(rewarded_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    del scorer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return len(rows)
