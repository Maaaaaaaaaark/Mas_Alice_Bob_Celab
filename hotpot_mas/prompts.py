"""Versioned prompt templates (spec sec. 11 and diagnostic conditions).

Prompt files live in standalone ``.txt`` files and are never spread across
Python modules. Each selected file is hashed (sha256), and the combined hash
feeds ``prompt_version``, so wording changes are detectable in the run records
(spec sec. 11.4). Templates use ``string.Template`` placeholders
(``$private_evidence``), which leaves all other characters (braces, dollars
in evidence text) untouched.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from string import Template
from typing import Dict

PROMPT_FILES = {
    "alice": "alice_system.txt",
    "bob": "bob_system.txt",
    "celab": "celab_system.txt",
    "forced_final": "forced_final_instruction.txt",
    "centralized_system": "centralized_system.txt",
    "centralized_task": "centralized_task.txt",
    "worker_task": "worker_task.txt",
    # Training-experiment prompt files (prompts_training/ directory).
    "worker_a_system": "worker_a_system.txt",
    "worker_b_system": "worker_b_system.txt",
    "synthesizer_system": "synthesizer_system.txt",
}

MAS_PROMPT_NAMES = ("alice", "bob", "celab", "forced_final")
CENTRALIZED_PROMPT_NAMES = ("centralized_system", "centralized_task")
ONE_SHOT_PROMPT_NAMES = MAS_PROMPT_NAMES + ("worker_task",)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PromptSet:
    """Load, hash, and render the templates selected by an experiment."""

    def __init__(self, prompt_dir: Path, names: tuple[str, ...] = MAS_PROMPT_NAMES):
        self.prompt_dir = Path(prompt_dir)
        self.templates: Dict[str, str] = {}
        self.hashes: Dict[str, str] = {}
        for name in names:
            filename = PROMPT_FILES[name]
            path = self.prompt_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"prompt template missing: {path}")
            text = path.read_text(encoding="utf-8").strip()
            self.templates[name] = text
            self.hashes[name] = _sha256(text)
        combined = "".join(self.hashes[name] for name in sorted(self.hashes))
        self.version = "v1-" + _sha256(combined)[:8]

    def render(self, name: str, **kwargs: str) -> str:
        """Render one template with its placeholders substituted."""
        return Template(self.templates[name]).safe_substitute(**kwargs)
