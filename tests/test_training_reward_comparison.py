from hotpot_mas.training.reward_comparison import summarize_reward_rows


def test_reward_summary_counts_signal_and_alignment():
    reports = {
        "a": [
            {"contains_normalized_gold": True},
            {"contains_normalized_gold": False},
        ],
        "b": [
            {"contains_normalized_gold": True},
            {"contains_normalized_gold": False},
        ],
    }
    matrix = [[1.0, 0.5], [0.5, 0.0]]
    row = {
        "reports": reports,
        "reward_matrices": {
            "em": [[1.0, 0.0], [0.0, 0.0]],
            "f1": matrix,
            "gold_mean_log_likelihood": matrix,
        },
    }
    summary = summarize_reward_rows([row], delta=0.05)
    assert summary["num_questions"] == 1
    assert summary["rewards"]["f1"]["questions_with_any_variance"] == 1
    assert summary["rewards"]["f1"]["report_contains_gold_correlation"] > 0
    assert summary["reward_ranking"][0] in {
        "em", "f1", "gold_mean_log_likelihood"
    }
