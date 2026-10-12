"""Data conversion for Stage 1a worker SFT (``docs/WORKER_SFT.md``).

Converts official rows into ``WorkerExample`` records:

- **SQuAD 2.0**: one example per row. Answerable rows become positives
  whose evidence is the sentence containing the official answer span;
  official unanswerable rows become ``NOT FOUND`` negatives.
- **MuSiQue**: one or more examples per answerable parent row, one per
  decomposition subquestion. ``#N`` placeholders are resolved to the
  answers of earlier subquestions (recorded in ``substitutions``); the
  evidence is the subquestion's support paragraph. A sibling paragraph
  that does not contain the answer is paired with the same subquestion as
  a negative.
- **HotpotQA**: examples come only from the reviewed (or optional
  teacher) files; there is no live teacher call. The full multi-hop
  question is recorded in ``original_question`` and never used as the
  local question.

The answer and its aliases live ONLY in the example metadata and are
never rendered into the prompt. Stable local document ids (``d0`` ...)
are stored here; display ids (``D0`` ...) are assigned at prompt-build
time after shuffling.

Original document text is stored untouched in ``documents[].text``;
``normalized_text`` is derived for matching only and never replaces it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Text helpers (documented, versioned)
# --------------------------------------------------------------------------

SPLITTER_VERSION = "worker-sft-v1"
# Split after . ! ? when followed by whitespace and a capital letter,
# a digit, or an opening quote. Documented heuristic; reviewed examples
# are verified against it and excluded on mismatch.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"'(0-9“])|"
    r"(?<=[.!?][\"”])\s+(?=[A-Z'(0-9“\"])"
)

# A sentence that cannot stand alone because it refers back to the
# previous sentence: prepend the previous sentence in auto candidates.
_PRONOUN_STARTS = (
    "it", "its", "he", "she", "they", "their", "this", "that",
    "these", "those", "his", "her",
)


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace. Matching only."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences with ``SPLITTER_VERSION``."""
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


def find_sentence_span(text: str, sentence: str) -> Optional[Tuple[int, int]]:
    """Character span of ``sentence`` inside ``text`` (exact match)."""
    start = text.find(sentence)
    if start < 0:
        return None
    return (start, start + len(sentence))


def sentence_index_of_offset(
    sentences: List[str], text: str, offset: int
) -> Optional[int]:
    """Index of the sentence containing character offset ``offset``."""
    position = 0
    for index, sentence in enumerate(sentences):
        # Rebuild the sentence span by searching for it after the current
        # position (handles repeated sentences correctly).
        start = text.find(sentence, position)
        if start < 0:
            return None
        if start <= offset < start + len(sentence):
            return index
        position = start + len(sentence)
    return None


def answer_offsets_in(
    text: str, answer: str
) -> Optional[Tuple[int, int]]:
    """First exact occurrence span of ``answer`` in ``text``, or None."""
    start = text.find(answer)
    if start < 0:
        return None
    return (start, start + len(answer))


def is_answer_in_text(
    text: str, answer: str, aliases: List[str]
) -> bool:
    """Basic (not sufficient) containment check for negative candidates."""
    normalized = normalize_text(text)
    candidates = [answer] + list(aliases)
    for candidate in candidates:
        normalized_candidate = normalize_text(candidate)
        if (
            normalized_candidate
            and f" {normalized_candidate} " in f" {normalized} "
        ):
            return True
    return False


# --------------------------------------------------------------------------
# The worker example
# --------------------------------------------------------------------------

@dataclass
class EvidenceSentence:
    doc_id: str
    sentence_index: int
    sentence: str
    char_span: Optional[Tuple[int, int]] = None


@dataclass
class WorkerDocument:
    doc_id: str
    title: str
    text: str  # original text, untouched
    normalized_text: str = ""


@dataclass
class WorkerExample:
    """One training/eval example: (local question, visible docs) -> target.

    ``answer``/``answer_aliases`` are metadata for checks and evaluation;
    they are never shown to the model. ``target`` is rendered at
    prompt-build time from ``evidence`` (or ``NOT FOUND`` when
    ``answerable`` is False).
    """

    example_id: str
    source: str  # squad2 | musique | hotpot
    split: str  # train | val | test (assigned by the pipeline)
    parent_id: str  # all variants of one parent share one split
    original_question: str  # full original question (never shown)
    local_question: str  # what the worker actually sees
    substitutions: List[Dict[str, str]]  # placeholder -> replacement record
    answer: str
    answer_aliases: List[str] = field(default_factory=list)
    answerable: bool = True
    documents: List[WorkerDocument] = field(default_factory=list)
    evidence: List[EvidenceSentence] = field(default_factory=list)
    negative_reason: Optional[str] = None
    probe: bool = False
    review: Dict[str, Any] = field(
        default_factory=lambda: {"status": "auto_checked"}
    )
    provenance: Dict[str, Any] = field(default_factory=dict)
    checks: Dict[str, Any] = field(default_factory=dict)
    excluded: bool = False
    exclusion_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "source": self.source,
            "split": self.split,
            "parent_id": self.parent_id,
            "original_question": self.original_question,
            "local_question": self.local_question,
            "substitutions": self.substitutions,
            "answer": self.answer,
            "answer_aliases": self.answer_aliases,
            "answerable": self.answerable,
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "text": doc.text,
                }
                for doc in self.documents
            ],
            "evidence": [
                {
                    "doc_id": ev.doc_id,
                    "sentence_index": ev.sentence_index,
                    "sentence": ev.sentence,
                    "char_span": list(ev.char_span)
                    if ev.char_span
                    else None,
                }
                for ev in self.evidence
            ],
            "negative_reason": self.negative_reason,
            "probe": self.probe,
            "review": self.review,
            "provenance": self.provenance,
            "checks": self.checks,
            "excluded": self.excluded,
            "exclusion_reason": self.exclusion_reason,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "WorkerExample":
        return cls(
            example_id=str(raw["example_id"]),
            source=str(raw["source"]),
            split=str(raw["split"]),
            parent_id=str(raw["parent_id"]),
            original_question=str(raw["original_question"]),
            local_question=str(raw["local_question"]),
            substitutions=[
                {str(k): str(v) for k, v in item.items()}
                for item in raw.get("substitutions", [])
            ],
            answer=str(raw.get("answer", "")),
            answer_aliases=[str(a) for a in raw.get("answer_aliases", [])],
            answerable=bool(raw["answerable"]),
            documents=[
                WorkerDocument(
                    doc_id=str(doc["doc_id"]),
                    title=str(doc["title"]),
                    text=str(doc["text"]),
                    normalized_text=normalize_text(str(doc["text"])),
                )
                for doc in raw.get("documents", [])
            ],
            evidence=[
                EvidenceSentence(
                    doc_id=str(ev["doc_id"]),
                    sentence_index=int(ev["sentence_index"]),
                    sentence=str(ev["sentence"]),
                    char_span=(
                        tuple(ev["char_span"])
                        if ev.get("char_span")
                        else None
                    ),
                )
                for ev in raw.get("evidence", [])
            ],
            negative_reason=(
                str(raw["negative_reason"])
                if raw.get("negative_reason")
                else None
            ),
            probe=bool(raw.get("probe", False)),
            review=dict(raw.get("review", {})),
            provenance=dict(raw.get("provenance", {})),
            checks=dict(raw.get("checks", {})),
            excluded=bool(raw.get("excluded", False)),
            exclusion_reason=(
                str(raw["exclusion_reason"])
                if raw.get("exclusion_reason")
                else None
            ),
        )

    def document_by_id(self, doc_id: str) -> WorkerDocument:
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        raise KeyError(f"unknown document id {doc_id!r} in {self.example_id}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc_from_musique_paragraph(
    index: int, paragraph: Dict[str, Any]
) -> WorkerDocument:
    doc = WorkerDocument(
        doc_id=f"d{index}",
        title=str(paragraph.get("title", "")),
        text=str(paragraph.get("paragraph_text", "")),
    )
    doc.normalized_text = normalize_text(doc.text)
    return doc


def _doc_from_hotpot(title: str, sentences: List[str]) -> WorkerDocument:
    text = " ".join(str(s) for s in sentences)
    doc = WorkerDocument(doc_id="d0", title=title, text=text)
    doc.normalized_text = normalize_text(doc.text)
    return doc


def _doc_from_squad(context: str) -> WorkerDocument:
    doc = WorkerDocument(doc_id="d0", title="", text=context)
    doc.normalized_text = normalize_text(doc.text)
    return doc


def _evidence_sentences_for_answer(
    doc: WorkerDocument,
    answer: str,
    answer_start: Optional[int] = None,
    include_second_for_pronoun: bool = True,
) -> Tuple[List[EvidenceSentence], Optional[str]]:
    """Locate evidence sentence(s) containing ``answer`` in ``doc``.

    Returns (evidence, reason_failed). In auto mode, when the located
    sentence starts with a pronoun the previous sentence is prepended so
    the target's subject/reference is identifiable.
    """
    sentences = split_sentences(doc.text)
    if answer_start is None:
        span = answer_offsets_in(doc.text, answer)
    else:
        end = answer_start + len(answer)
        span = (answer_start, end) if doc.text[answer_start:end] == answer else None
    if span is None:
        return [], f"answer {answer!r} not found verbatim in {doc.doc_id}"
    index = sentence_index_of_offset(sentences, doc.text, span[0])
    if index is None:
        return [], "answer offset outside any sentence"
    indices = [index]
    first = sentences[index].lower()
    if (
        include_second_for_pronoun
        and index > 0
        and first.split(" ", 1)[0].strip(".,!?;:") in _PRONOUN_STARTS
    ):
        indices = [index - 1, index]
    evidence = []
    for sentence_index in indices:
        sentence = sentences[sentence_index]
        evidence.append(
            EvidenceSentence(
                doc_id=doc.doc_id,
                sentence_index=sentence_index,
                sentence=sentence,
                char_span=find_sentence_span(doc.text, sentence),
            )
        )
    return evidence, None


# --------------------------------------------------------------------------
# Official-row converters
# --------------------------------------------------------------------------

def convert_squad2_row(
    row: Dict[str, Any], source_meta: Dict[str, Any]
) -> List[WorkerExample]:
    """One example per SQuAD 2.0 row (positive or official negative)."""
    row_id = str(row["id"])
    question = str(row["question"])
    context = str(row["context"])
    answers = row.get("answers") or {"text": [], "answer_start": []}
    answerable = bool(answers.get("text"))
    example_id = f"squad2-{row_id}"
    doc = _doc_from_squad(context)
    provenance = dict(source_meta)
    provenance["row_id"] = row_id
    if answerable:
        answer = str(answers["text"][0])
        answer_start = int(answers["answer_start"][0])
        evidence, failed = _evidence_sentences_for_answer(
            doc, answer, answer_start=answer_start
        )
        example = WorkerExample(
            example_id=example_id,
            source="squad2",
            split="train",
            parent_id=row_id,
            original_question=question,
            local_question=question,
            substitutions=[],
            answer=answer,
            answer_aliases=[],
            answerable=True,
            documents=[doc],
            evidence=evidence,
            provenance=provenance,
        )
        example.provenance["answer_offsets"] = (
            answer_start,
            answer_start + len(answer),
        )
        if failed:
            example.excluded = True
            example.exclusion_reason = f"squad2 conversion: {failed}"
        return [example]
    example = WorkerExample(
        example_id=example_id,
        source="squad2",
        split="train",
        parent_id=row_id,
        original_question=question,
        local_question=question,
        substitutions=[],
        answer="",
        answerable=False,
        documents=[doc],
        negative_reason=(
            "official SQuAD 2.0 unanswerable example (empty answer list)"
        ),
        provenance=provenance,
    )
    return [example]


def resolve_subquestion(
    subquestion: str, prior_answers: List[str]
) -> Tuple[str, List[Dict[str, str]]]:
    """Replace ``#N`` placeholders with earlier subquestion answers.

    ``#N`` refers to the answer of subquestion ``N-1``. The current
    subquestion's own answer is never inserted (only earlier answers
    exist in ``prior_answers``). Returns (resolved, substitutions).
    """
    substitutions: List[Dict[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        if index < 0 or index >= len(prior_answers):
            # Unresolvable reference: leave the placeholder so the check
            # can flag it, and record the failure.
            return match.group(0)
        placeholder = match.group(0)
        replacement = prior_answers[index]
        substitutions.append(
            {
                "placeholder": placeholder,
                "replacement": replacement,
                "source": f"answer of subquestion {index}",
            }
        )
        return replacement

    resolved = re.sub(r"#(\d+)", replace, subquestion)
    return resolved, substitutions


_UNRESOLVED_REF_RE = re.compile(r"#\d+")


def convert_musique_row(
    row: Dict[str, Any],
    source_meta: Dict[str, Any],
    max_negatives_per_parent: int = 1,
) -> List[WorkerExample]:
    """Examples from one MuSiQue parent: subquestions + sibling negatives.

    Every decomposition subquestion of an answerable row becomes a
    positive example whose visible document is its support paragraph.
    Additionally, up to ``max_negatives_per_parent`` subquestions are
    paired with a sibling paragraph that does not contain the answer.
    All examples of a parent share ``parent_id`` so they land in one
    split.
    """
    parent_id = str(row["id"])
    decomposition = row.get("question_decomposition") or []
    paragraphs = row.get("paragraphs") or []
    provenance = dict(source_meta)
    provenance["row_id"] = parent_id
    examples: List[WorkerExample] = []
    prior_answers: List[str] = []
    resolved_questions: List[str] = []
    substitutions_by_index: Dict[int, List[Dict[str, str]]] = {}

    for index, sub in enumerate(decomposition):
        subquestion = str(sub.get("question", ""))
        sub_answer = str(sub.get("answer", ""))
        support_idx = sub.get("paragraph_support_idx")
        resolved, substitutions = resolve_subquestion(subquestion, prior_answers)
        resolved_questions.append(resolved)
        substitutions_by_index[index] = substitutions
        sub_provenance = dict(provenance)
        sub_provenance["subquestion_index"] = index
        sub_provenance["original_subquestion"] = subquestion
        sub_provenance["paragraph_support_idx"] = support_idx

        if support_idx is None or support_idx >= len(paragraphs):
            prior_answers.append(sub_answer)
            continue
        support = paragraphs[support_idx]
        doc = _doc_from_musique_paragraph(support_idx, support)
        evidence, failed = _evidence_sentences_for_answer(doc, sub_answer)
        example = WorkerExample(
            example_id=f"musique-{parent_id}-sub{index}",
            source="musique",
            split="train",
            parent_id=parent_id,
            original_question=str(row.get("question", "")),
            local_question=resolved,
            substitutions=substitutions,
            answer=sub_answer,
            answer_aliases=[],
            answerable=True,
            documents=[doc],
            evidence=evidence,
            provenance=sub_provenance,
        )
        example.provenance["paragraph_title"] = doc.title
        if failed:
            example.excluded = True
            example.exclusion_reason = f"musique conversion: {failed}"
        if _UNRESOLVED_REF_RE.search(resolved):
            example.excluded = True
            example.exclusion_reason = (
                "musique conversion: unresolved #N reference in local question"
            )
        examples.append(example)
        prior_answers.append(sub_answer)

    # Negatives: re-ask a subquestion against a sibling paragraph that
    # does not contain the answer (a basic check, not proof of
    # unanswerability; recorded as such in the label-quality report).
    negatives_made = 0
    support_indices = {
        sub.get("paragraph_support_idx")
        for sub in decomposition
        if sub.get("paragraph_support_idx") is not None
    }
    for index, sub in enumerate(decomposition):
        if negatives_made >= max_negatives_per_parent:
            break
        sub_answer = str(sub.get("answer", ""))
        if not sub_answer:
            continue
        sibling = None
        for p_index, paragraph in enumerate(paragraphs):
            # A negative may not reuse any gold support paragraph from
            # the same decomposition: another hop's support can still
            # contain enough relational evidence to answer indirectly.
            if p_index in support_indices:
                continue
            if not is_answer_in_text(
                str(paragraph.get("paragraph_text", "")),
                sub_answer,
                [],
            ):
                sibling = (p_index, paragraph)
                break
        if sibling is None:
            continue
        p_index, paragraph = sibling
        sibling_doc = _doc_from_musique_paragraph(p_index, paragraph)
        sub_provenance = dict(provenance)
        sub_provenance["subquestion_index"] = index
        sub_provenance["original_subquestion"] = str(sub.get("question", ""))
        sub_provenance["paragraph_support_idx"] = sub.get(
            "paragraph_support_idx"
        )
        sub_provenance["negative_sibling_paragraph_idx"] = p_index
        example = WorkerExample(
            example_id=f"musique-{parent_id}-sub{index}-neg{p_index}",
            source="musique",
            split="train",
            parent_id=parent_id,
            original_question=str(row.get("question", "")),
            local_question=resolved_questions[index],
            substitutions=substitutions_by_index.get(index, []),
            answer=sub_answer,
            answer_aliases=[],
            answerable=False,
            documents=[sibling_doc],
            negative_reason=(
                "sibling paragraph of the same MuSiQue parent that does "
                "not contain the answer string (basic check, not proof)"
            ),
            provenance=sub_provenance,
        )
        examples.append(example)
        negatives_made += 1
    return examples


# --------------------------------------------------------------------------
# Reviewed / teacher entries (HotpotQA and verified examples)
# --------------------------------------------------------------------------

def _locate_reviewed_evidence(
    doc: WorkerDocument,
    evidence_texts: List[str],
    answer: str,
    aliases: List[str],
) -> Tuple[List[EvidenceSentence], Optional[str]]:
    """Verify stored evidence sentences and derive offsets/indexes.

    Each stored sentence must occur verbatim in the document and match
    one or two consecutive sentences under the splitter. For positives
    the evidence must contain the answer or a verified alias.
    """
    sentences = split_sentences(doc.text)
    located: List[EvidenceSentence] = []
    for evidence_text in evidence_texts:
        span = find_sentence_span(doc.text, evidence_text)
        if span is None:
            return [], (
                f"evidence sentence not found verbatim in {doc.doc_id}: "
                f"{evidence_text[:80]!r}"
            )
        # The evidence must align with sentence boundaries.
        start_offset = span[0]
        running = 0
        sentence_index = None
        for index, sentence in enumerate(sentences):
            sentence_start = doc.text.find(sentence, running)
            if sentence_start < 0:
                return [], "splitter produced an unreproducible sentence"
            if sentence_start == start_offset:
                sentence_index = index
                break
            running = sentence_start + len(sentence)
        if sentence_index is None:
            return [], (
                "evidence text is not aligned with a sentence boundary "
                f"in {doc.doc_id}: {evidence_text[:80]!r}"
            )
        located.append(
            EvidenceSentence(
                doc_id=doc.doc_id,
                sentence_index=sentence_index,
                sentence=evidence_text,
                char_span=span,
            )
        )
    if len(located) > 2:
        return [], "more than two evidence sentences"
    if len(located) == 2 and (
        located[0].doc_id != located[1].doc_id
        or located[1].sentence_index != located[0].sentence_index + 1
    ):
        return [], "two evidence sentences must be consecutive in one document"
    if answer and not is_answer_in_text(
        " ".join(ev.sentence for ev in located), answer, aliases
    ):
        return [], (
            "evidence does not contain the answer or a verified alias"
        )
    return located, None


def resolve_reviewed_entry(
    entry: Dict[str, Any],
    loaders: Dict[str, Any],
) -> Tuple[Optional[WorkerExample], Optional[str]]:
    """Build a verified example from one reviewed-file entry.

    ``loaders`` maps source -> loader function that returns the official
    row (or raises KeyError). Any mismatch between the reviewed entry and
    the official row excludes the example with a recorded reason; nothing
    is silently corrected.
    """
    example_id = str(entry["example_id"])
    source = str(entry["source"])
    pointer = entry.get("pointer") or {}
    loader = loaders.get(source)
    if loader is None:
        return None, f"no loader for source {source!r}"

    try:
        row = loader(pointer)
    except KeyError:
        return None, (
            f"official row not found: {pointer!r} (source {source})"
        )
    except Exception as exc:  # noqa: BLE001 - recorded, never silent
        return None, f"failed to load official row: {exc}"

    expected_row_id = pointer.get("row_id")
    actual_row_id = row.get("id")
    if expected_row_id is not None and str(actual_row_id) != str(expected_row_id):
        return None, (
            f"official row mismatch: requested {expected_row_id!r}, "
            f"loader returned {actual_row_id!r}"
        )

    review = dict(entry.get("review") or {})
    review.setdefault("status", "reviewed")
    review.setdefault("reviewer", entry.get("reviewed_by", "unknown"))
    review["provenance_file"] = "data/worker_sft_reviewed.json"

    documents: List[WorkerDocument] = []
    if source == "squad2":
        context = str(row["context"])
        doc = _doc_from_squad(context)
        documents = [doc]
        row_id = str(row["id"])
    elif source == "musique":
        sub_index = int(pointer["subquestion_index"])
        decomposition = row.get("question_decomposition") or []
        if sub_index >= len(decomposition):
            return None, (
                f"subquestion_index {sub_index} out of range for "
                f"{pointer.get('row_id')!r}"
            )
        sub = decomposition[sub_index]
        support_idx = sub.get("paragraph_support_idx")
        paragraphs = row.get("paragraphs") or []
        titles = [str(p.get("title", "")) for p in paragraphs]
        # The reviewed entry lists titles; the visible document is the
        # entry's first document title (support or swapped sibling).
        entry_titles = [str(d["title"]) for d in entry.get("documents", [])]
        if not entry_titles:
            return None, "reviewed musique entry lists no documents"
        visible_title = entry_titles[0]
        p_index = None
        for idx, title in enumerate(titles):
            if title == visible_title:
                p_index = idx
                break
        if p_index is None:
            return None, (
                f"visible document title {visible_title!r} not found in row "
                f"{pointer.get('row_id')!r}"
            )
        doc = _doc_from_musique_paragraph(p_index, paragraphs[p_index])
        documents = [doc]
        row_id = str(row["id"])
    elif source == "hotpot":
        from hotpot_mas.hotpotqa import row_context_documents

        pointer_titles = list(pointer.get("titles", []))
        if row.get("context") is not None:
            context_documents = row_context_documents(row)
            row_titles = [str(doc["title"]) for doc in context_documents]
            row_sentences = [doc["sentences"] for doc in context_documents]
        else:
            # Compatibility with small legacy/local fixtures.
            row_titles = [str(t) for t in row.get("title", [])]
            row_sentences = row.get("sentences", [])
        entry_titles = [str(d["title"]) for d in entry.get("documents", [])]
        visible_titles = entry_titles or pointer_titles
        for doc_index, title in enumerate(visible_titles):
            if title not in row_titles:
                return None, (
                    f"document title {title!r} not found in hotpot row "
                    f"{pointer.get('row_id')!r}"
                )
            row_index = row_titles.index(title)
            sentences = row_sentences[row_index]
            if not isinstance(sentences, list):
                sentences = [str(sentences)]
            documents.append(_doc_from_hotpot(title, sentences))
            # Stable ids come from the entry when present (multi-doc
            # probes declare their own order).
            entry_docs = entry.get("documents") or []
            if entry_docs and len(entry_docs) > doc_index:
                documents[-1].doc_id = str(entry_docs[doc_index]["doc_id"])
            else:
                documents[-1].doc_id = f"d{doc_index}"
        row_id = str(row["id"])
    else:
        return None, f"unknown reviewed source {source!r}"

    # Verify the local question when the entry points at a decomposition
    # subquestion (MuSiQue): the reviewed question must match the
    # official resolution.
    if source == "musique":
        sub_index = int(pointer["subquestion_index"])
        decomposition = row.get("question_decomposition") or []
        prior_answers = [
            str(s.get("answer", "")) for s in decomposition[:sub_index]
        ]
        resolved, _ = resolve_subquestion(
            str(decomposition[sub_index].get("question", "")), prior_answers
        )
        entry_question = str(entry.get("local_question", ""))
        if resolved != entry_question:
            return None, (
                f"reviewed local_question {entry_question!r} does not match "
                f"the official resolution {resolved!r}"
            )

    # Verify evidence sentences.
    evidence_entry = entry.get("evidence") or []
    if not bool(entry.get("answerable", True)) and evidence_entry:
        return None, f"{example_id}: unanswerable entry must not have evidence"
    evidence_by_doc: Dict[str, List[str]] = {}
    for ev in evidence_entry:
        evidence_by_doc.setdefault(str(ev["doc_id"]), []).append(
            str(ev["sentence"])
        )
    located: List[EvidenceSentence] = []
    for doc in documents:
        texts = evidence_by_doc.get(doc.doc_id, [])
        answer = str(entry.get("answer", ""))
        if entry.get("answerable", True) and answer:
            located_here, failed = _locate_reviewed_evidence(
                doc, texts, answer, list(entry.get("answer_aliases", []))
            )
        else:
            located_here = []
            for text in texts:
                span = find_sentence_span(doc.text, text)
                if span is None:
                    return None, (
                        f"evidence sentence not found verbatim in "
                        f"{doc.doc_id}: {text[:80]!r}"
                    )
            failed = None
        if failed:
            return None, f"{example_id}: {failed}"
        located.extend(located_here)

    answerable = bool(entry.get("answerable", True))
    if answerable and not located:
        return None, f"{example_id}: answerable entry has no evidence"
    if not answerable and located:
        return None, (
            f"{example_id}: unanswerable entry must not have evidence"
        )

    example = WorkerExample(
        example_id=example_id,
        source=source,
        split=str(entry.get("split", "train")),
        parent_id=(
            str(entry.get("parent_id")) if entry.get("parent_id") else row_id
        ),
        original_question=str(entry.get("original_question", "")),
        local_question=str(entry.get("local_question", "")),
        substitutions=[
            {str(k): str(v) for k, v in s.items()}
            for s in entry.get("substitutions", [])
        ],
        answer=str(entry.get("answer", "")),
        answer_aliases=[str(a) for a in entry.get("answer_aliases", [])],
        answerable=answerable,
        documents=documents,
        evidence=located,
        negative_reason=(
            str(entry["negative_reason"]) if entry.get("negative_reason") else None
        ),
        probe=bool(entry.get("probe", False)),
        review=review,
        provenance={
            "dataset": {
                "squad2": "squad_v2",
                "musique": "musique_ans_v1.0",
                "hotpot": "hotpot_qa",
            }[source],
            "config": {
                "squad2": "default",
                "musique": "default",
                "hotpot": "distractor",
            }[source],
            "split": str(entry.get("split", "train")),
            "row_id": row_id,
            "pointer": pointer,
            "reviewed": True,
        },
    )
    return example, None


def load_reviewed_file(
    path: Any,
    loaders: Dict[str, Any],
) -> Tuple[List[WorkerExample], List[Dict[str, str]]]:
    """Load and resolve all reviewed entries.

    Returns (examples, problems): problems is a list of
    {example_id, reason} for entries that could not be resolved; the
    caller reports them explicitly instead of silently dropping them.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if raw.get("splitter_version") != SPLITTER_VERSION:
        raise ValueError(
            f"reviewed file splitter version {raw.get('splitter_version')!r} "
            f"does not match this code's {SPLITTER_VERSION!r}"
        )
    examples: List[WorkerExample] = []
    problems: List[Dict[str, str]] = []
    for entry in raw.get("entries", []):
        example, problem = resolve_reviewed_entry(entry, loaders)
        if problem is not None:
            problems.append(
                {"example_id": str(entry.get("example_id", "?")), "reason": problem}
            )
            continue
        assert example is not None
        examples.append(example)
    return examples, problems


def load_teacher_file(path: Any) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Load an optional teacher-generated local-question file.

    Returns (entries, status). The status is ``{"status": "missing"}``
    when the file does not exist; the caller must report it. Entries are
    NOT auto-added to training: the pipeline resolves them against
    official rows and applies the same checks and review gate.
    """
    if not path.is_file():
        return [], {"status": "missing", "path": str(path)}
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries, {"status": "loaded", "path": str(path), "count": len(entries)}
