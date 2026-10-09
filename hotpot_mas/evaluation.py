"""HotpotQA answer evaluation (spec sec. 13).

``normalize_answer``, ``f1_score``, and ``exact_match_score`` are a verbatim
(logic-identical) port of the official HotpotQA evaluator:

    https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py

Normalization: lowercase, strip punctuation, remove the articles a/an/the,
fix whitespace. F1 is token-level over whitespace tokens with a yes/no/
noanswer guard; EM is normalized equality. HotpotQA validation gold answers
are single strings; the list branch in ``evaluate_answer`` only exists for
robustness and takes the max over gold answers, matching common practice in
official baselines.

The reference implementation in ``references/reward.py::cal_f1_score``
tokenizes with the LLM's own tokenizer and skips the official normalization;
it is deliberately NOT used here.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import List, Tuple, Union


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Official token-level F1 between a prediction and one gold answer."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0.0
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Official EM: normalized equality with one gold answer."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_overlap_scores(
    prediction: str, ground_truth: str
) -> Tuple[float, float, float]:
    """Return (precision, recall, f1) for one prediction/gold pair.

    Same normalization, token overlap, and yes/no/noanswer guard as the
    official evaluator, but exposes the precision and recall components so
    training traces can record them alongside the F1 reward.
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0.0, 0.0, 0.0
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0.0, 0.0, 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def evaluate_answer(
    prediction: Union[str, None], gold: Union[str, List[str]]
) -> Tuple[float, float]:
    """Return (f1, em) for a prediction against one or more gold answers.

    ``prediction`` must already be the extracted ``<FINAL>...</FINAL>`` inner
    text; ``None`` or an empty string is scored as an empty prediction
    (spec sec. 7).
    """
    if prediction is None:
        prediction = ""
    golds = gold if isinstance(gold, (list, tuple)) else [gold]
    f1 = max(f1_score(prediction, g) for g in golds)
    em = max(exact_match_score(prediction, g) for g in golds)
    return f1, em
