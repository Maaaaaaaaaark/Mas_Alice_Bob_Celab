"""Static checks for the Violet environment-activated Slurm workflow."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def test_every_slurm_script_activates_the_pinned_environment_before_python():
    for path in sorted(SCRIPTS.glob("*.sbatch")):
        text = path.read_text(encoding="utf-8")
        assert "module load Anaconda3/2024.06-1" in text
        assert 'conda activate "$ENV_DIR"' in text
        assert '[[ "$CONDA_PREFIX" == "$ENV_DIR" ]]' in text
        python_position = text.find("srun python")
        if python_position >= 0:
            assert text.find('conda activate "$ENV_DIR"') < python_position
        assert "CUDA_VISIBLE_DEVICES" not in text


def test_full_arrays_use_four_question_shards_and_one_a5000_each():
    for filename in (
        "violet_full_mas_array.sbatch",
        "violet_full_single_array.sbatch",
    ):
        text = (SCRIPTS / filename).read_text(encoding="utf-8")
        assert "#SBATCH --array=0-3%4" in text
        assert "#SBATCH --gres=gpu:1" in text
        assert "#SBATCH --constraint=a5000" in text
        assert "NUM_SHARDS=4" in text
        assert '--shard-index "$SLURM_ARRAY_TASK_ID"' in text


def test_merge_job_requests_no_gpu_and_merges_both_conditions():
    text = (SCRIPTS / "violet_merge_reports.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres" not in text
    assert text.count("python -m hotpot_mas.cli merge-shards") == 2
    assert "one_shot_direct_answer_distractor.yaml" in text
    assert "single_agent_distractor.yaml" in text
