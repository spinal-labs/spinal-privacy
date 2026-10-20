"""Shared helpers for the frozen Qwen3.8-27B target.

Model loading, the Alpaca corpus, greedy generation, the clean-versus-protected agreement metric,
nearest-neighbour decoding, and the output-folder and report handling shared by the scripts.
Importing this module needs neither a GPU nor the model.

The model folder defaults to models/Qwen3.8-27B in the repository, or QWEN_MODEL_DIR when set.
Download the pinned revision, for example:
    hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
        --local-dir models/Qwen3.8-27B
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# Set before transformers is imported: the Hugging Face libraries read these at import time.
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

MODEL_ID = 'Qwen/Qwen3.8-27B'
MODEL_REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
MODEL_CONFIG = {
    'hidden_size': 5120,
    'vocab_size': 248320,
    'num_hidden_layers': 64,
    'max_position_embeddings': 262144,
}
MODEL = Path(os.environ.get('QWEN_MODEL_DIR', REPO / 'models' / 'Qwen3.8-27B'))
DATA = REPO / 'data' / 'alpaca.jsonl'
START = time.monotonic()
EVENTS: list[dict[str, Any]] = []


def event(kind: str, **values: Any) -> dict[str, Any]:
    """Record a progress event in EVENTS and print it as one JSON line."""
    row = {'event': kind, 'elapsed_seconds': time.monotonic() - START, **values}
    EVENTS.append(row)
    print(json.dumps(row), flush=True)
    return row


def seed(value: int) -> None:
    """Seed the Python, CPU and CUDA random number generators."""
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def check_model(path: Path) -> bool:
    """Warn when the model folder does not look like the pinned target. Warns only; never stops a run."""
    try:
        config = json.loads((Path(path) / 'config.json').read_text())
    except (OSError, ValueError):
        warnings.warn(
            f'cannot read {path}/config.json; expected {MODEL_ID} at revision {MODEL_REVISION}', stacklevel=2
        )
        return False
    text = config.get('text_config', config)
    wrong = {k: text.get(k) for k, v in MODEL_CONFIG.items() if text.get(k) != v}
    if wrong:
        warnings.warn(
            f'model config differs from {MODEL_ID} at revision {MODEL_REVISION}: {wrong}', stacklevel=2
        )
    return not wrong


def load(model_dir: Path) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load the tokenizer and the frozen bf16 model from a local folder onto the GPU."""
    check_model(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation='sdpa',
            device_map='cuda',
        )
        .eval()
        .requires_grad_(False)
    )
    model.config.use_cache = False
    return tok, model


def corpus(tok: PreTrainedTokenizerBase) -> list[dict[str, Any]]:
    """Alpaca prompts without an input field, tokenized with thinking disabled.

    Each record holds the prompt text, the reference response, the prompt token ids and a content
    mask that is False on special tokens. Special tokens are preserved.
    """
    records = []
    for line in DATA.read_text().splitlines():
        row = json.loads(line)
        if row.get('input'):
            continue
        text = tok.apply_chat_template(
            [{'role': 'user', 'content': row['text']}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tok(text, add_special_tokens=False)['input_ids']
        records.append(
            {
                'id': f"alpaca-{len(records):04d}",
                'prompt': row['text'],
                'reference': row['response'],
                'prompt_ids': ids,
                'content_mask': [i not in tok.all_special_ids for i in ids],
            }
        )
    return records


@torch.no_grad()
def generate(model: PreTrainedModel, z: torch.Tensor, eos: set[int], limit: int = 64) -> list[int]:
    """Greedy decoding from inputs_embeds with the KV cache on. z is [1, seq, dim]."""
    embedding = model.get_input_embeddings()
    dtype = next(model.parameters()).dtype
    past, step, out = None, z.to(dtype), []
    for _ in range(limit):
        result = model(inputs_embeds=step, past_key_values=past, use_cache=True)
        past = result.past_key_values
        token = int(result.logits[:, -1].argmax(-1))
        out.append(token)
        if token in eos:
            break
        step = embedding(torch.tensor([[token]], device=z.device)).to(dtype)
    return out


def agreement(clean_ids: list[int], protected_ids: list[int]) -> dict[str, Any]:
    """Clean-vs-protected output agreement over the GENERATED PREFIX only.

    Generation stops at the token cap or EOS, so "no divergence" means "no divergence
    observed within the generated output", not agreement beyond it.

    first_divergence is None when nothing diverged. It is never replaced by the sequence
    length: that would conflate "never failed" with "failed at the last token" and make an
    all-exact run report its mean output length. The longest common prefix is a valid
    descriptive number and is reported separately as longest_common_prefix.
    """
    n = min(len(clean_ids), len(protected_ids))
    matched = sum(a == b for a, b in zip(clean_ids[:n], protected_ids[:n]))
    first = next((i for i, (a, b) in enumerate(zip(clean_ids, protected_ids)) if a != b), None)
    kind = 'token_substitution' if first is not None else None
    if first is None and len(clean_ids) != len(protected_ids):
        first, kind = n, 'prefix_exhaustion'  # identical prefix, one ran on or stopped early
    return {
        'exact': clean_ids == protected_ids,
        'token_agreement': matched / max(len(clean_ids), len(protected_ids), 1),
        'longest_common_prefix': n if first is None else first,
        'first_divergence': first,
        'first_divergence_kind': kind,
        'diverged': first is not None,
        'clean_len': len(clean_ids),
        'protected_len': len(protected_ids),
        'length_delta': len(protected_ids) - len(clean_ids),
    }


def divergence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Divergence rate, and where divergence starts among pairs that actually diverge.

    Rejects rows without a 'diverged' field: treating a missing field as False would
    silently report a 0% divergence rate.

    The conditional mean is composition-sensitive -- rescuing failures changes which
    prompts remain in the average -- so the rate and the count are reported alongside it.
    """
    if not rows:
        return {'divergence_rate_pct': None, 'n_total': 0, 'note': 'no observations'}
    missing = [i for i, r in enumerate(rows) if 'diverged' not in r]
    if missing:
        raise ValueError(f"{len(missing)} rows carry no 'diverged' field; build rows with agreement()")
    for r in rows:
        if r['diverged'] == r['exact']:
            raise ValueError(f"inconsistent row {r.get('id')}: diverged={r['diverged']} exact={r['exact']}")
    positions = [r['first_divergence'] for r in rows if r['diverged']]
    positions_sorted = sorted(positions)

    def quantile(q: float) -> int | None:
        return (
            positions_sorted[min(int(q * len(positions_sorted)), len(positions_sorted) - 1)]
            if positions_sorted
            else None
        )

    return {
        'divergence_rate_pct': 100.0 * len(positions) / len(rows),
        'n_diverged': len(positions),
        'n_total': len(rows),
        'first_divergence_mean_when_diverged': (sum(positions) / len(positions)) if positions else None,
        'first_divergence_median_when_diverged': quantile(0.5),
        'first_divergence_p10_when_diverged': quantile(0.1),
        'first_divergence_p90_when_diverged': quantile(0.9),
        'diverged_at_first_token_pct': 100.0 * sum(p == 0 for p in positions) / len(rows),
    }


@torch.no_grad()
def nearest(observed: torch.Tensor, table: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Nearest vocabulary row by cosine similarity, for every vector in observed."""
    flat = F.normalize(observed.float().reshape(-1, observed.shape[-1]), dim=-1)
    best = flat.new_full((len(flat),), -float('inf'))
    ids = torch.zeros(len(flat), dtype=torch.long, device=flat.device)
    for off in range(0, len(table), chunk):
        cand = F.normalize(table[off : off + chunk].float(), dim=-1)
        score, local = (flat @ cand.T).max(-1)
        ids = torch.where(score > best, local + off, ids)
        best = torch.maximum(score, best)
    return ids.reshape(observed.shape[:-1])


def refuse_nonempty_output(parser: argparse.ArgumentParser, out: Path, overwrite: bool) -> None:
    """Stop with a usage error when out is not a folder, or already holds files and overwrite is off."""
    if out.exists() and not out.is_dir():
        parser.error(f'--out {out} exists and is not a folder')
    if not overwrite and out.is_dir() and any(out.iterdir()):
        parser.error(f'--out {out} is not empty; pass --overwrite to write into it')


def run_with_report(
    main: Callable[[argparse.Namespace], None], args: argparse.Namespace, report: dict[str, Any]
) -> int:
    """Run main(args) and always write args.out/report.json with the report and the events.

    On failure the report records the traceback, which is also printed to stderr. Returns the exit
    status: 0 on success, 1 on failure.
    """
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        main(args)
    except Exception:
        report['status'] = 'error'
        report['error'] = traceback.format_exc()
        print(report['error'], file=sys.stderr, flush=True)
        return 1
    finally:
        (args.out / 'report.json').write_text(json.dumps({**report, 'events': EVENTS}, indent=2) + '\n')
    return 0
