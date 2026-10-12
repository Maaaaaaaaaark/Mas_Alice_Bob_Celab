"""Selection tests for worker-SFT probes, weak negatives, and source weights."""

from hotpot_mas.training.worker_sft.checks import checked_example
from hotpot_mas.training.worker_sft.mix import eligible_train, select_smoke_or_normal

from worker_sft_test_utils import (
    build_cpu_config,
    make_doc,
    make_evidence,
    make_example,
)


def _item(name, source, answerable, reviewed=True, probe=False):
    answer = "blue" if answerable else "missing"
    sentence = "The object is blue."
    example = make_example(
        name,
        f"Question {name}?",
        [make_doc("d0", sentence if answerable else "No fact.")],
        answer=answer,
        answerable=answerable,
        evidence=[make_evidence("d0", sentence)] if answerable else [],
        negative_reason=None if answerable else "negative fixture",
        source=source,
        review_status="reviewed" if reviewed else "auto_checked",
        probe=probe,
    )
    checked_example(example)
    return example


def test_eligible_train_excludes_probes_and_unreviewed_musique_negatives():
    good = _item("good", "squad2", True)
    probe = _item("probe", "squad2", True, probe=True)
    weak_negative = _item("weak", "musique", False, reviewed=False)
    assert eligible_train([good, probe, weak_negative], False) == [good]


def test_equal_source_weights_are_applied_to_both_labels(tmp_path):
    cfg = build_cpu_config(tmp_path)
    cfg.mix.total_examples = 6
    cfg.mix.target_answerable_ratio = 0.5
    examples = []
    for source in ("squad2", "musique", "hotpot"):
        examples.append(_item(f"{source}-p", source, True))
        examples.append(_item(f"{source}-n", source, False))
    selected, report = select_smoke_or_normal(examples, cfg, False)
    assert len(selected) == 6
    assert {
        source: sum(item.source == source for item in selected)
        for source in ("squad2", "musique", "hotpot")
    } == {"squad2": 2, "musique": 2, "hotpot": 2}
    assert report["source_weights"] == {
        "squad2": 1.0,
        "musique": 1.0,
        "hotpot": 1.0,
    }
