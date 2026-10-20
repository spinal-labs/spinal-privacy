"""Run identity, independent controls, and frozen format validation for the MMLU protocol."""

import hashlib
import importlib.metadata
import json
import math
import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

FORMAT_TOKENS = 32
FORMAT_SEED = 20260911


def file_sha(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def object_sha(value: Any) -> str:
    """SHA-256 of the canonical JSON form of `value`."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    """Write canonical JSON atomically (temp file, fsync, rename)."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def freeze_manifest(path: str | Path, value: Any) -> None:
    """Write once; on later calls require the same content.

    The comparison is made on the JSON canonical form of `value`, not on the in-memory
    object: the real model config carries integer dictionary keys (`id2label: {0: ...}`)
    that JSON renders as strings, so an in-memory comparison against the parsed file
    would reject every genuine resume.
    """
    path = Path(path)
    canonical = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    if path.exists():
        if json.loads(path.read_text()) != canonical:
            raise ValueError(f'{path}: existing manifest differs; use a new output directory')
    else:
        write_json(path, value)


def system_identity(model_dir: str | Path, tok: Any, model: Any, source_paths: Iterable[str | Path]) -> dict:
    """Hash actual local weights/config/tokenizer bytes, not a mutable directory name."""
    root = Path(model_dir)
    assets = {
        str(p.relative_to(root)): file_sha(p)
        for p in sorted(root.rglob('*'))
        if p.is_file()
        and p.suffix in {'.json', '.safetensors', '.bin', '.model', '.txt', '.jinja', '.py', '.tiktoken'}
    }
    if not assets or not any(k.endswith(('.safetensors', '.bin')) for k in assets):
        raise ValueError('model identity requires local weight files')
    versions = {}
    for package in ('torch', 'transformers', 'tokenizers', 'accelerate', 'safetensors'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        'assets': assets,
        'versions': versions,
        'model_class': type(model).__name__,
        'tokenizer_class': type(tok).__name__,
        'model_config': model.config.to_dict(),
        'generation_config': model.generation_config.to_dict(),
        'chat_template': tok.chat_template,
        'special_ids': tok.all_special_ids,
        'tokenizer_backend_sha': object_sha(tok.backend_tokenizer.to_str()),
        'model_dtype': str(next(model.parameters()).dtype),
        'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
        'sources': {str(p): file_sha(p) for p in source_paths},
    }


def validate_items(items: list[dict], split: str) -> list[dict]:
    """Check that every item is a well-formed four-choice question with a unique id."""
    if not items:
        raise ValueError('empty split')
    ids = set()
    for r in items:
        for field in ('question', 'subject'):
            if not isinstance(r.get(field), str) or not r[field].strip():
                raise ValueError(f'invalid {field}')
        if (
            not isinstance(r.get('choices'), list)
            or len(r['choices']) != 4
            or not all(isinstance(c, str) and c.strip() for c in r['choices'])
        ):
            raise ValueError('expected four nonempty string choices')
        for field in ('i', 'answer'):
            if type(r.get(field)) is not int or r[field] < 0:
                raise ValueError(f'invalid integral {field}')
        if r['answer'] > 3:
            raise ValueError('gold outside 0..3')
        ident = item_id(r, split)
        if ident in ids:
            raise ValueError(f'duplicate source item {ident}')
        ids.add(ident)
    return items


def item_id(row: dict, split: str = 'mmlu_dev') -> str:
    """Stable identifier of a source item."""
    return f"{split}:{row['subject']}:{row['i']}"


def select_format_items(items: list[dict]) -> list[dict]:
    """Five items per subject, chosen by a seeded hash so the choice is reproducible."""
    subjects = sorted({r['subject'] for r in items})
    if len(subjects) != 57:
        raise ValueError('format validation requires all 57 subjects')
    selected = []
    for subject in subjects:
        group = sorted(
            (r for r in items if r['subject'] == subject), key=lambda r: object_sha([FORMAT_SEED, item_id(r)])
        )
        if len(group) < 5:
            raise ValueError(f'{subject}: fewer than five format items')
        selected.extend(group[:5])
    return selected


def select_validation_items(
    items: list[dict], excluded_ids: Iterable[str], per_subject: int = 5
) -> list[dict]:
    """Format-validation subset: the `select_format_items` ordering, minus excluded items.

    Items are ordered per subject by ascending SHA-256 of [FORMAT_SEED, item_id]. Every id in
    `excluded_ids` (the subset on which the prompt format was chosen) is skipped, then the first
    `per_subject` survivors are kept. A subject with fewer survivors contributes what it has,
    possibly none; the count is recorded, never padded.
    """
    excluded = set(excluded_ids)
    if not excluded:
        raise ValueError('an exclusion set is required; use select_format_items otherwise')
    subjects = sorted({r['subject'] for r in items})
    if len(subjects) != 57:
        raise ValueError('format validation requires all 57 subjects in the source split')
    selected = []
    for subject in subjects:
        group = sorted(
            (r for r in items if r['subject'] == subject and item_id(r) not in excluded),
            key=lambda r: object_sha([FORMAT_SEED, item_id(r)]),
        )
        selected.extend(group[:per_subject])
    if not selected:
        raise ValueError('no validation items survive the exclusion')
    if any(item_id(r) in excluded for r in selected):
        raise AssertionError('validation subset overlaps the excluded set')
    return selected


def parse_answer(text: str) -> int | None:
    """Only a complete bare uppercase label, optionally followed by one period.

    Whitespace is allowed at either end. Explanations, multiple labels, empty and
    truncated generations are failures, never dropped from the denominator.
    """
    match = re.fullmatch(r'\s*([ABCD])\.?\s*', text)
    return 'ABCD'.index(match[1]) if match else None


def assert_scores(
    actual: Sequence[float], expected: Sequence[float], label: str = 'control', tolerance: float = 1e-9
) -> None:
    """Raise unless two score vectors agree within `tolerance` and pick the same option."""
    from evaluation.mmlu_scorer import predict

    if (
        len(actual) != 4
        or len(expected) != 4
        or any(not math.isfinite(x) for x in list(actual) + list(expected))
    ):
        raise AssertionError(f'{label}: invalid scores')
    if (
        max(abs(a - b) for a, b in zip(actual, expected)) > tolerance
        or predict(actual)[0] != predict(expected)[0]
    ):
        raise AssertionError(f'{label}: scores/predictions disagree: {actual} vs {expected}')


class BoundaryProbe:
    """Observe the first decoder block after outer dispatch and forward pre-hooks.

    This verifies prompt delivery to the decoder, not arbitrary downstream internals.
    Unknown architectures fail closed. A synthetic non-identity prompt is also tested
    by the runner, so alpha=0 does not make the dispatch control vacuous.
    """

    def __init__(self, model: torch.nn.Module, boundary: torch.nn.Module | None = None) -> None:
        self.model = model
        if boundary is None:
            candidates = [
                module[0]
                for name, module in model.named_modules()
                if name.endswith('layers') and isinstance(module, torch.nn.ModuleList) and len(module)
            ]
            if len(candidates) != 1:
                raise ValueError('C1c: cannot uniquely locate the first decoder block')
            boundary = candidates[0]
        self.expected = None
        self.last = None
        self.calls = 0
        self.handle = boundary.register_forward_pre_hook(self._observe, with_kwargs=True)

    def _observe(self, module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
        if self.expected is None:
            return
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        if hidden is None:
            raise AssertionError('C1c: decoder received no hidden_states')
        self.last = hidden.detach().clone()
        self.calls += 1
        n = self.expected.shape[1]
        if (
            hidden.dtype != self.expected.dtype
            or hidden.shape[0] != 1
            or hidden.shape[1] < n
            or hidden.shape[2:] != self.expected.shape[2:]
            or not torch.equal(hidden[:, :n], self.expected)
        ):
            raise AssertionError('C1c: first decoder block did not receive the expected prompt')

    def expect(self, prompt: torch.Tensor) -> None:
        """Arm the probe: the next forward pass must deliver `prompt` to the first block."""
        self.expected = prompt.detach().clone()
        self.last = None
        self.calls = 0

    def saw(self, expected: torch.Tensor, *, must_differ_from: torch.Tensor | None = None) -> bool:
        """Check what the first block received on the last call."""
        if not self.calls or self.last is None:
            raise AssertionError('C1c: decoder boundary was never reached')
        prefix = self.last[:, : expected.shape[1]]
        if prefix.dtype != expected.dtype or not torch.equal(prefix, expected):
            raise AssertionError('C1c: unexpected decoder input')
        if must_differ_from is not None and torch.equal(prefix, must_differ_from):
            raise AssertionError('C1c: non-identity control is identical to clean')
        return True

    def __call__(self, **kwargs: Any) -> Any:
        return self.model(**kwargs)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.get_input_embeddings()

    def close(self) -> None:
        """Remove the hook."""
        self.handle.remove()


@torch.no_grad()
def oracle_scores(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    label_ids: list[list[int]],
    embeds: torch.Tensor | None = None,
) -> list[float]:
    """Independent uncached teacher forcing; clean ids or real mixed prompt.

    Calls forward directly to bypass outer __call__ dispatch/pre-hooks. Does not call
    the production scorer, share its indexing code, or share a KV cache.
    """
    device = next(model.parameters()).device
    answers = []
    for candidate in label_ids:
        if embeds is None:
            sequence = torch.tensor([list(prompt_ids) + candidate[:-1]], device=device)
            logits = model.forward(input_ids=sequence, use_cache=False).logits[0]
        else:
            tail = model.get_input_embeddings()(
                torch.tensor([candidate[:-1]], device=device, dtype=torch.long)
            )
            sequence = torch.cat([embeds, tail.to(embeds.dtype)], 1)
            logits = model.forward(inputs_embeds=sequence, use_cache=False).logits[0]
        if len(candidate) == 1:
            value = float(logits[len(prompt_ids) - 1, candidate[0]])
        else:
            value = 0.0
            for j, token in enumerate(candidate):
                vector = logits[len(prompt_ids) - 1 + j].double()
                value += float(vector[token] - torch.logsumexp(vector, dim=0))
        answers.append(value)
    return answers


def eos_ids(model: Any, tok: Any) -> set[int]:
    """All end-of-sequence token ids named by the model's generation config and the tokenizer."""
    ids = model.generation_config.eos_token_id
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    if not ids:
        raise ValueError('no EOS token configured')
    return set(ids)
