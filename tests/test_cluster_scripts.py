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


def test_every_slurm_script_uses_the_submission_directory_as_repo_root():
    for path in sorted(SCRIPTS.glob("*.sbatch")):
        text = path.read_text(encoding="utf-8")
        assert 'REPO_DIR="${SLURM_SUBMIT_DIR:?' in text
        assert "/projects/Mas_Alice_Bob_Celab" not in text


def test_full_arrays_use_four_question_shards_and_one_a5000_each():
    for filename in (
        "violet_full_mas_array.sbatch",
        "violet_full_single_array.sbatch",
    ):
        text = (SCRIPTS / filename).read_text(encoding="utf-8")
        assert "#SBATCH --array=0-3%4" in text
        assert "#SBATCH --gres=gpu:1" in text
        assert "#SBATCH --constraint=a5000" in text
        assert "#SBATCH --partition=researchlong" in text
        assert "#SBATCH --qos=research-1-qos" in text
        assert "NUM_SHARDS=4" in text
        assert '--shard-index "$SLURM_ARRAY_TASK_ID"' in text


def test_smoke_jobs_use_short_research_partition():
    for filename in (
        "violet_smoke_mas.sbatch",
        "violet_smoke_single.sbatch",
    ):
        text = (SCRIPTS / filename).read_text(encoding="utf-8")
        assert "#SBATCH --partition=researchshort" in text
        assert "#SBATCH --qos=research-1-qos" in text


def test_merge_job_requests_no_gpu_and_merges_both_conditions():
    text = (SCRIPTS / "violet_merge_reports.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres" not in text
    assert text.count("python -m hotpot_mas.cli merge-shards") == 2
    assert "one_shot_direct_answer_distractor.yaml" in text
    assert "single_agent_distractor.yaml" in text


def test_failure_analysis_job_is_cpu_only_and_calls_failure_cli():
    text = (SCRIPTS / "violet_analyze_mas_failures.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres" not in text
    assert "python -m hotpot_mas.cli analyze-failures" in text
    assert '--runs "$RUNS_PATH"' in text
    assert '--output-dir "$OUTPUT_DIR"' in text


def test_worker_sft_job_preps_data_then_trains_on_one_a5000():
    text = (SCRIPTS / "violet_worker_sft.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --constraint=a5000" in text
    assert "#SBATCH --partition=researchshort" in text
    assert "#SBATCH --qos=research-1-qos" in text
    assert "#SBATCH --account=pradeepresearch" in text
    assert 'MODE="${WORKER_SFT_MODE:-trace}"' in text
    assert (
        "srun python -m hotpot_mas.training.cli prepare-worker-sft-data"
        in text
    )
    assert (
        "srun python -m hotpot_mas.training.cli train-worker-sft" in text
    )
    # Optional rebuild/resume switches stay opt-in, not defaults.
    assert 'WORKER_SFT_FORCE_DATA:-0' in text
    assert 'WORKER_SFT_RESUME:-0' in text
    # The pinned HF cache and the exact trace-mode output paths are
    # printed by the job so the researcher knows where artifacts land.
    assert 'HF_HOME="$MAS_SCRATCH/hf_cache"' in text
    assert "worked_example_worker_sft.md" in text
    assert "configs/worker_sft.yaml" in text
    # No silent GPU assumptions: the job itself checks what it got.
    assert "nvidia-smi" in text
    assert "srun whichgpu" in text
    # Nonzero exit codes propagate (set -euo pipefail + no swallowing).
    assert "set -euo pipefail" in text
