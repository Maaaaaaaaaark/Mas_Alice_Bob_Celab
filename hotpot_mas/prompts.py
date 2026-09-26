"""Versioned prompt templates (spec sec. 11).

The three system prompts and the fixed forced-final instruction live in
standalone ``.txt`` files (``prompts/``) and are never spread across Python
modules. Each file is hashed (sha256) and the combined hash feeds
``prompt_version``, so any wording change is detectable in the run records
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
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PromptSet:
    """Loads, hashes, and renders the four versioned prompt templates."""

    def __init__(self, prompt_dir: Path):
        self.prompt_dir = Path(prompt_dir)
        self.templates: Dict[str, str] = {}
        self.hashes: Dict[str, str] = {}
        for name, filename in PROMPT_FILES.items():
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
