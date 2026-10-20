"""Paired clean-vs-protected MMLU utility measurement under the scoring protocol (PROTOCOL.md).

The split manifest check and format validation precede measurement. Only mmlu_dev is permitted.
C1 runs on every item; C1a/C1c use independent teacher forcing on each session's
first 20 new items, including a synthetic non-identity dispatch control at alpha=0.

Every row also carries five diagnostic fields -- the clean winner's signed protected lead and
its change from the clean lead, the mean relative embedding displacement, and the
per-content-token nearest-neighbour cosine margins in fp32 and bf16. They are part of
ItemResult and therefore of the protocol hash; no scored quantity depends on them. Rows saved
under a different protocol hash are not resumed.

An existing output folder is resumed when its manifest matches this run: saved rows are validated and
reused, and only missing items are measured. A folder with measurements but no manifest is refused.

Usage:  python scripts/evaluate.py --checkpoint runs/train/fixed_fp64.pt [--alpha 1.0] [--model-dir DIR]
                                   [--data-dir DIR] [--out DIR] [--limit N]
Writes: OUT/manifest.json, format_rows.jsonl, format_report.json, rows.jsonl, report.json
        (OUT defaults to runs/mmlu-a<alpha>)
"""

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch  # noqa: E402

import common  # noqa: E402
from evaluation.mmlu_contract import (  # noqa: E402
    FORMAT_TOKENS,
    BoundaryProbe,
    assert_scores,
    eos_ids,
    file_sha,
    freeze_manifest,
    item_id,
    object_sha,
    oracle_scores,
    parse_answer,
    select_format_items,
    select_validation_items,
    system_identity,
    validate_items,
    write_json,
)
from evaluation.mmlu_scorer import (  # noqa: E402
    ItemResult,
    clean_winner_lead,
    diagnose_format,
    diagnostics_summary,
    label_token_ids,
    obfuscation_summary,
    predict,
    protocol_hash,
    render,
    score_options,
    summarise,
    transition,
    validate,
)
from privacy_transform.model import PrivacyTransform, TransformConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO / 'data'
SPLIT = 'mmlu_dev'  # the only split this script measures
SEED = 20260911
# Prompt format (PROTOCOL.md).
# INSTRUCTION is the optional line inside the user message; PREFILL is appended to the chat
# template's assistant generation prefix as text BEFORE tokenization, exactly as probed.
INSTRUCTION = 'Respond with only the letter of the correct option: A, B, C, or D.'
PREFILL = 'Answer:'
FORMAT_FILE = 'mmlu_format_validation_r5.jsonl'
FORMAT_EXCLUDED_FILE = 'mmlu_format_validation.jsonl'  # subset the prompt format was chosen on; excluded


def prompt_text(tok: Any, question: str, choices: list[str]) -> str:
    """The complete prompt string: user message through chat template, then the prefill."""
    text = tok.apply_chat_template(
        [{'role': 'user', 'content': render(question, choices, INSTRUCTION, answer_line=False)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(text, str) or not text:
        raise ValueError('chat template returned no text')
    return text + PREFILL


def build_prompt(tok: Any, question: str, choices: list[str]) -> tuple[list[int], list[bool]]:
    """Prompt token ids and the content mask (False on special tokens)."""
    ids = tok(prompt_text(tok, question, choices), add_special_tokens=False)['input_ids']
    if not ids:
        raise ValueError('empty tokenized prompt')
    return ids, [i not in tok.all_special_ids for i in ids]


def blend(e: torch.Tensor, t: torch.Tensor, alpha: float) -> torch.Tensor:
    """The delivered embeddings (1 - alpha) * E + alpha * T(E), with exact checks at both endpoints."""
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be finite and in [0,1]')
    if not torch.isfinite(e).all() or not torch.isfinite(t).all():
        raise ValueError('non-finite transform output')
    z = (1 - alpha) * e + alpha * t
    if alpha == 0 and not torch.equal(z, e):
        raise AssertionError('C2: alpha=0 differs from E')
    if alpha == 1 and not torch.equal(z, t):
        raise AssertionError('C2: alpha=1 differs from T(E)')
    return z


def obfuscation(
    z: torch.Tensor, ids: list[int], content: list[bool], table: torch.Tensor
) -> tuple[int, int, int]:
    """Changed content tokens under nearest-neighbour decoding in fp32 and in bf16, and the content count."""
    p32 = common.nearest(z[0], table).tolist()
    p16 = common.nearest(z[0].to(torch.bfloat16).float(), table).tolist()
    return (
        sum(a != b for a, b, k in zip(p32, ids, content) if k),
        sum(a != b for a, b, k in zip(p16, ids, content) if k),
        sum(content),
    )


@torch.no_grad()
def lookup_margins(
    z: torch.Tensor, ids: list[int], content: list[bool], table: torch.Tensor, chunk: int = 4096
) -> tuple[list[float], list[float]]:
    """Per content token: cosine with its own row minus the best cosine with any other row.

    fp32 normalised cosine over the whole table, so a negative margin means the
    nearest-neighbour decoder returns another token and a zero margin is a tie. The
    changed-token COUNT stays with the pinned decoder (`common.nearest`); validate() checks
    the two agree up to a small tolerance.
    Returns (fp32 margins, bf16-cast margins) as plain floats.
    """
    keep = torch.tensor(content, device=z.device)
    own_ids = torch.tensor(ids, device=z.device)[keep]
    if not len(own_ids):
        return [], []
    out = []
    for vectors in (z[0][keep].float(), z[0][keep].to(torch.bfloat16).float()):
        flat = torch.nn.functional.normalize(vectors, dim=-1)
        best_other = flat.new_full((len(flat),), -float('inf'))
        own = flat.new_full((len(flat),), float('nan'))
        for off in range(0, len(table), chunk):
            cand = torch.nn.functional.normalize(table[off : off + chunk].float(), dim=-1)
            score = flat @ cand.T
            local = own_ids - off
            inside = (local >= 0) & (local < score.shape[1])
            if inside.any():
                rows = inside.nonzero().flatten()
                own[rows] = score[rows, local[rows]]
                score[rows, local[rows]] = -float('inf')
            best_other = torch.maximum(best_other, score.max(-1).values)
        if not torch.isfinite(own).all() or not torch.isfinite(best_other).all():
            raise ValueError('lookup margin: own or rival cosine is not finite')
        out.append((own - best_other).tolist())
    return out[0], out[1]


def displacement_mean(z32: torch.Tensor, e32: torch.Tensor, content: list[bool]) -> float:
    """Mean over content tokens of ||z_t - e_t|| / ||e_t||, fp32."""
    keep = torch.tensor(content, device=z32.device)
    if not keep.any():
        return 0.0
    z, e = z32[0][keep].float(), e32[0][keep].float()
    value = ((z - e).norm(dim=-1) / e.norm(dim=-1).clamp_min(1e-12)).mean()
    if not torch.isfinite(value):
        raise ValueError('non-finite displacement')
    return float(value)


def restore_rows(
    path: Path,
    phash: str,
    items: list[dict[str, Any]],
    prompts: dict[str, tuple[list[int], list[bool]]],
) -> list[dict[str, Any]]:
    """Reject foreign IDs, wrong gold/subject, incomplete controls and corrupt diagnostics."""
    if not path.exists():
        return []
    expected = {item_id(it): it for it in items}

    def check(saved: dict[str, Any]) -> dict[str, Any]:
        checksum = saved.pop('row_sha', None)
        if checksum != object_sha(saved) or saved['protocol_hash'] != phash:
            raise ValueError('resume row checksum or protocol mismatch')
        raw = saved['result']
        r = ItemResult(**raw)
        if r.item_id not in expected:
            raise ValueError(f'resume row outside requested selection: {r.item_id}')
        source = expected[r.item_id]
        ids, content = prompts[r.item_id]
        if (
            r.gold != source['answer']
            or r.subject != source['subject']
            or r.prompt_tokens != len(ids)
            or r.content_tokens != sum(content)
        ):
            raise ValueError('resume row disagrees with source/prompt')
        if not all(type(v) is bool for v in saved['controls'].values()) or not all(
            saved['controls'].get(k) is True for k in ('C1', 'C1c', 'C2')
        ):
            raise ValueError('resume row lacks required controls')
        for arm in ('clean', 'protected'):
            lp = saved[arm + '_logprobs']
            if (
                len(lp) != 4
                or not all(math.isfinite(x) and x <= 0 for x in lp)
                or abs(sum(math.exp(x) for x in lp) - getattr(r, arm + '_mass')) > 1e-12
            ):
                raise ValueError('resume log-probabilities contradict candidate mass')
            # The ranking scores and the log-probabilities must describe ONE distribution.
            # Single-token: logprob_i = score_i - logZ, a constant offset. Multi-token: equal.
            # Without this a row could carry protected log-probabilities favouring B beside
            # ranking scores favouring A, each internally consistent, and validate() would
            # accept the A prediction.
            offsets = [x - y for x, y in zip(lp, getattr(r, arm + '_scores'))]
            if max(offsets) - min(offsets) > 1e-6:
                raise ValueError(
                    f'{r.item_id}: {arm} log-probabilities and ranking scores '
                    'do not describe the same distribution'
                )
        return saved

    raw_bytes = path.read_bytes()
    complete, fragment = raw_bytes, b''
    if raw_bytes and not raw_bytes.endswith(b'\n'):
        keep = raw_bytes.rfind(b'\n') + 1
        complete, fragment = raw_bytes[:keep], raw_bytes[keep:]
    rows = [check(json.loads(line)) for line in complete.decode().splitlines() if line.strip()]
    if fragment:
        # A run can be interrupted mid-write (for example by a reboot). A row is written as
        # one line plus newline, so an interrupted write leaves an unterminated fragment.
        # Newline termination and row validity are separate properties: a complete row whose
        # newline alone was lost is kept and its terminator restored; anything that does not
        # parse or fails its checksum is discarded and the file truncated in place, so the
        # next append cannot fuse with it. Interior corruption still raises above. The file
        # is only modified after every complete line has been accepted.
        try:
            candidate = json.loads(fragment)
            intact = isinstance(candidate, dict) and candidate.get('row_sha') == object_sha(
                {k: v for k, v in candidate.items() if k != 'row_sha'}
            )
        except ValueError:
            intact = False
        if intact:
            with path.open('ab') as handle:
                handle.write(b'\n')
                handle.flush()
                os.fsync(handle.fileno())
            rows.append(check(candidate))
            print(json.dumps({'event': 'restored_missing_newline', 'path': str(path)}), flush=True)
        else:
            print(
                json.dumps(
                    {'event': 'truncated_partial_row', 'path': str(path), 'dropped_bytes': len(fragment)}
                ),
                flush=True,
            )
            with path.open('r+b') as handle:
                handle.truncate(len(complete))
    validate([ItemResult(**r['result']) for r in rows])
    return rows


def reconcile(
    saved_rows: list[dict[str, Any]], format_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """An item saved in both rows.jsonl and format_rows.jsonl must be the same measurement.

    Format validation is judged on format_rows.jsonl and the regression rate on rows.jsonl.
    Two individually valid but different records for one item would let each file support
    a different claim. Generation text is only in the format file and is not compared.
    """
    for r in saved_rows:
        f = format_by_id.get(r['result']['item_id'])
        if f is None:
            continue
        for key in ('result', 'clean_logprobs', 'protected_logprobs', 'controls'):
            if f[key] != r[key]:
                raise ValueError(
                    f"{r['result']['item_id']}: rows.jsonl and format_rows.jsonl disagree on {key}"
                )
    return saved_rows


@torch.no_grad()
def generate_format(
    model: Any, probe: BoundaryProbe, z: torch.Tensor, eos: set[int], tok: Any
) -> dict[str, Any]:
    """Greedy free generation for format validation, watched by the boundary probe."""
    step, past, tokens = z, None, []
    for _ in range(FORMAT_TOKENS):
        probe.expect(step)
        out = model(inputs_embeds=step, past_key_values=past, use_cache=True)
        probe.saw(step)
        vector = out.logits[0, -1]
        if not torch.isfinite(vector).all():
            raise ValueError('non-finite generation logits')
        token = int(vector.argmax())
        tokens.append(token)
        past = out.past_key_values
        if token in eos:
            break
        step = model.get_input_embeddings()(torch.tensor([[token]], device=z.device))
    ended = tokens[-1] in eos
    text = tok.decode(tokens, skip_special_tokens=True)
    return {
        'token_ids': tokens,
        'text': text,
        'ended': ended,
        'parsed': parse_answer(text) if ended else None,
    }


@torch.no_grad()
def main() -> None:
    """Measure clean and protected MMLU predictions for every selected item and write the report."""
    parser = argparse.ArgumentParser(
        description='Paired MMLU utility measurement (needs a CUDA GPU and the model).'
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='transform checkpoint: a .pt file from scripts/train.py, or a folder with model.safetensors '
        'and config.json',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='blend strength in [0, 1]: the model receives (1 - alpha) * E + alpha * T(E) (default: 1.0)',
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        default=common.MODEL,
        help='local Qwen3.8-27B folder (default: $QWEN_MODEL_DIR or models/Qwen3.8-27B)',
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=DATA_DIR,
        help='folder with mmlu_dev.jsonl, manifest.json and the format-validation files (default: data/)',
    )
    parser.add_argument(
        '--out',
        type=Path,
        help='output folder, resumed when its manifest matches (default: runs/mmlu-a<alpha>)',
    )
    parser.add_argument(
        '--limit', type=int, default=0, help='measure only the first N items, 0 for all (default: 0)'
    )
    args = parser.parse_args()
    if not math.isfinite(args.alpha) or not 0 <= args.alpha <= 1 or args.limit < 0:
        raise ValueError('require alpha in [0,1] and non-negative limit')
    out_dir = args.out or REPO / 'runs' / f'mmlu-a{args.alpha}'
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.data_dir / 'mmlu_dev.jsonl'
    all_items = validate_items([json.loads(line) for line in split_path.read_text().splitlines()], SPLIT)
    split_manifest = json.loads((args.data_dir / 'manifest.json').read_text())
    if file_sha(split_path) != split_manifest['splits']['mmlu_dev']['sha256']:
        raise ValueError('split differs from frozen split manifest')
    items = all_items[: args.limit] if args.limit else all_items
    excluded_path = args.data_dir / FORMAT_EXCLUDED_FILE
    excluded_items = [json.loads(line) for line in excluded_path.read_text().splitlines()]
    if excluded_items != select_format_items(all_items):
        raise ValueError('excluded format subset differs from its deterministic selection')
    format_items = select_validation_items(all_items, {item_id(it) for it in excluded_items})
    format_path = args.data_dir / FORMAT_FILE
    if [json.loads(line) for line in format_path.read_text().splitlines()] != format_items:
        raise ValueError('format subset differs from deterministic selection')
    torch.backends.cuda.matmul.allow_tf32 = False
    common.seed(SEED)
    tok, model = common.load(args.model_dir)
    if model.training or next(model.parameters()).dtype != torch.bfloat16:
        raise AssertionError('model must be eval and bf16')
    embedding = model.get_input_embeddings()
    table = embedding.weight.detach()
    label_ids, label_len = label_token_ids(tok)
    state = common.load_checkpoint(args.checkpoint)
    config = TransformConfig(**state['config'])
    trained = PrivacyTransform(config).cuda().eval().requires_grad_(False)
    trained.load_state_dict(state['state_dict'])
    identity = PrivacyTransform(config).cuda().eval().requires_grad_(False)
    prompts = {item_id(it): build_prompt(tok, it['question'], it['choices']) for it in all_items}
    sources = [
        Path(__file__),
        REPO / 'evaluation/mmlu_scorer.py',
        REPO / 'evaluation/mmlu_contract.py',
        Path(getattr(common, '__file__', ROOT / 'common.py')),
        REPO / 'privacy_transform/model.py',
        REPO / 'PROTOCOL.md',
    ]
    contract = {
        'revision': 5,
        'amendment': 'diagnostic fields',
        'instruction': INSTRUCTION,
        'prefill': PREFILL,
        'format_rule': 'format_validation ordering minus the format_validation subset, up to 5 per subject',
        'format_excluded_sha': file_sha(excluded_path),
        'checkpoint_sha': file_sha(state['weights_file']),
        'transform_config': state['config'],
        'alpha': args.alpha,
        'seed': SEED,
        'split_sha': file_sha(split_path),
        'split_manifest_sha': file_sha(args.data_dir / 'manifest.json'),
        'selection_ids': [item_id(it) for it in items],
        'format_ids': [item_id(it) for it in format_items],
        'format_sha': file_sha(format_path),
        'format_budget': FORMAT_TOKENS,
        'label_ids': label_ids,
        'label_len': label_len,
        'system': system_identity(args.model_dir, tok, model, sources),
        'rendered_prefix': prompt_text(tok, items[0]['question'], items[0]['choices']),
        'prompt_ids_sha': object_sha(prompts),
        'precision': 'model bf16, transform fp32',
        'tf32': False,
        'scorer_protocol_hash': protocol_hash(),
    }
    phash = object_sha(contract)
    manifest_path = out_dir / 'manifest.json'
    if not manifest_path.exists() and any(
        (out_dir / n).exists() for n in ('rows.jsonl', 'format_rows.jsonl', 'report.json')
    ):
        raise ValueError('existing measurements have no manifest; use a new output directory')
    freeze_manifest(manifest_path, {'protocol_hash': phash, **contract})
    cfg = getattr(model.config, 'text_config', model.config)
    max_context = cfg.max_position_embeddings
    format_ids = {item_id(it) for it in format_items}
    # +2 covers the two-token C1a fixture on extended-control items.
    required = {
        item_id(it): len(prompts[item_id(it)][0])
        + (max(label_len, FORMAT_TOKENS) if item_id(it) in format_ids else max(label_len, 2))
        for it in items + format_items
    }
    exclusions = [
        {'item_id': ident, 'prompt_tokens': len(prompts[ident][0]), 'required_tokens': length}
        for ident, length in required.items()
        if length > max_context
    ]
    if exclusions:
        write_json(out_dir / 'exclusions.json', exclusions)
        raise ValueError('context limit exceeded; no truncation allowed')
    probe = BoundaryProbe(model)
    controls = {
        'C1': 0,
        'C1a': 0,
        'C1a_multi': 0,
        'C1c': 0,
        'C1c_oracle': 0,
        'C1c_nonidentity': 0,
        'C2': 0,
        'C3': 0,
    }
    eos = eos_ids(model, tok)
    # C1a includes a multi-token fixture even though MMLU labels are expected to be
    # single-token: four distinct two-token continuations built from the real label ids
    # plus one EOS id, scored on the target through both paths.
    multi_fixture = [list(label) + [min(eos)] for label in label_ids]

    def measure(it: dict[str, Any], extended: bool) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
        """Score one item clean and protected with its controls; extended adds the teacher-forcing ones."""
        ids, content = prompts[item_id(it)]
        e = embedding(torch.tensor([ids], device='cuda'))
        e32 = e.float()
        attn = torch.ones(e.shape[:2], dtype=torch.bool, device=e.device)
        cmask = torch.tensor([content], device=e.device)
        probe.expect(e)
        cs, clp, cma = score_options(probe, e.clone(), label_ids)
        probe.saw(e)
        zi32 = identity(e32, attn, cmask, stochastic=False)
        if not torch.equal(zi32, e32):
            raise AssertionError('C1: fresh deterministic transform is not identity')
        zi = zi32.to(e.dtype)
        probe.expect(zi)
        isc, ilp, _ = score_options(probe, zi.clone(), label_ids)
        probe.saw(zi)
        assert_scores(isc, cs, 'C1')
        assert_scores(ilp, clp, 'C1 logprobs')
        controls['C1'] += 1
        t = trained(e32, attn, cmask, stochastic=False)
        blend(e32, t, 0.0)
        blend(e32, t, 1.0)
        controls['C2'] += 1
        z32 = blend(e32, t, args.alpha)
        z = z32.to(e.dtype)
        probe.expect(z)
        ps, plp, pma = score_options(probe, z.clone(), label_ids)
        probe.saw(z)
        controls['C1c'] += 1
        if extended:
            for endpoint, direct, reference in ((0.0, e, cs), (1.0, t.to(e.dtype), None)):
                sent = blend(e32, t, endpoint).to(e.dtype)
                probe.expect(sent)
                endpoint_scores, _, _ = score_options(probe, sent.clone(), label_ids)
                probe.saw(sent)
                if reference is None:
                    probe.expect(direct)
                    reference = oracle_scores(model, ids, label_ids, direct.clone())
                    probe.saw(direct)
                assert_scores(endpoint_scores, reference, f'C2 alpha={endpoint}')
            probe.expect(e)
            assert_scores(cs, oracle_scores(model, ids, label_ids), 'C1a')
            probe.saw(e)
            controls['C1a'] += 1
            probe.expect(e)
            multi_scores, _, _ = score_options(probe, e.clone(), multi_fixture)
            assert_scores(multi_scores, oracle_scores(model, ids, multi_fixture), 'C1a multi-token')
            probe.saw(e)
            controls['C1a_multi'] += 1
            probe.expect(z)
            assert_scores(ps, oracle_scores(model, ids, label_ids, z.clone()), 'C1c oracle')
            probe.saw(z)
            controls['C1c_oracle'] += 1
            synthetic = e.clone()
            synthetic[:, 0] = synthetic[:, 0] + 1
            probe.expect(synthetic)
            synth, _, _ = score_options(probe, synthetic.clone(), label_ids)
            probe.saw(synthetic, must_differ_from=e)
            probe.expect(synthetic)
            assert_scores(synth, oracle_scores(model, ids, label_ids, synthetic.clone()), 'C1c synthetic')
            controls['C1c_nonidentity'] += 1
            a, b, _ = obfuscation(e32, ids, content, table)
            if a or b:
                raise AssertionError('C3: decoder fails on clean fp32/bf16 input')
            controls['C3'] += 1
        ch32, ch16, total = obfuscation(z32, ids, content, table)
        m32, m16 = lookup_margins(z32, ids, content, table)
        lead, lead_change = clean_winner_lead(cs, ps)
        cp, ct, cm = predict(cs)
        pp, pt, pm = predict(ps)
        gold = it['answer']
        row = ItemResult(
            item_id(it),
            it['subject'],
            gold,
            cp,
            pp,
            cs,
            ps,
            cp == gold,
            pp == gold,
            transition(cp == gold, pp == gold),
            cma,
            pma,
            cm,
            pm,
            ct,
            pt,
            ch32,
            ch16,
            total,
            len(ids),
            cp != pp,
            clean_winner_protected_lead=lead,
            clean_winner_lead_change=lead_change,
            displacement_mean=displacement_mean(z32, e32, content),
            lookup_margins=m32,
            lookup_margins_bf16=m16,
        )
        validate([row])
        saved = {
            'protocol_hash': phash,
            'result': asdict(row),
            'clean_logprobs': clp,
            'protected_logprobs': plp,
            'controls': {
                'C1': True,
                'C1c': True,
                'C2': True,
                'C1a': extended,
                'C1a_multi': extended,
                'C1c_oracle': extended,
                'C1c_nonidentity': extended,
                'C3': extended,
            },
        }
        return saved, e, z

    def append(path: Path, saved: dict[str, Any]) -> None:
        """Append one checksummed row and flush it to disk."""
        with path.open('a') as sink:
            sink.write(json.dumps({**saved, 'row_sha': object_sha(saved)}, allow_nan=False) + '\n')
            sink.flush()
            os.fsync(sink.fileno())

    try:
        # Format validation is completed first, including on limited smoke runs.
        fp = out_dir / 'format_rows.jsonl'
        format_saved = restore_rows(fp, phash, format_items, prompts)
        done = {r['result']['item_id'] for r in format_saved}
        fresh = 0
        for it in format_items:
            if item_id(it) in done:
                continue
            saved, e, z = measure(it, fresh < 20)
            saved['generation'] = {
                arm: generate_format(model, probe, emb, eos, tok)
                for arm, emb in (('clean', e), ('protected', z))
            }
            append(fp, saved)
            format_saved.append(saved)
            fresh += 1
            if fresh % 20 == 0:
                common.event('format_progress', done=len(format_saved), total=len(format_items))
        agreement = {}
        for arm in ('clean', 'protected'):
            matches = 0
            for r in format_saved:
                g = r['generation'][arm]
                text = tok.decode(g['token_ids'], skip_special_tokens=True)
                ended = bool(g['token_ids']) and g['token_ids'][-1] in eos
                parsed = parse_answer(text) if ended else None
                if text != g['text'] or ended != g['ended'] or parsed != g['parsed']:
                    raise ValueError('cached format parse is inconsistent')
                matches += parsed == r['result'][arm + '_pred']
            if len(format_saved) != len(format_items):
                raise AssertionError('format validation is incomplete')
            agreement[arm] = matches / len(format_saved)
        format_diag = diagnose_format([ItemResult(**r['result']) for r in format_saved], agreement)
        write_json(out_dir / 'format_report.json', {'protocol_hash': phash, **format_diag})
        # Reuse validated subset scores so they are not measured twice.
        rows_path = out_dir / 'rows.jsonl'
        saved_rows = restore_rows(rows_path, phash, items, prompts)
        done = {r['result']['item_id'] for r in saved_rows}
        format_by_id = {r['result']['item_id']: r for r in format_saved}
        reconcile(saved_rows, format_by_id)
        fresh = 0
        for it in items:
            ident = item_id(it)
            if ident in done:
                continue
            if ident in format_by_id:
                saved = format_by_id[ident]
            else:
                saved, _, _ = measure(it, fresh < 20)
                fresh += 1
            append(rows_path, saved)
            saved_rows.append(saved)
            done.add(ident)
            if len(saved_rows) % 50 == 0:
                common.event('progress', done=len(saved_rows), total=len(items))
        if done != {item_id(it) for it in items}:
            raise AssertionError('incomplete measurement coverage')
        combined = format_saved + saved_rows
        aggregate_controls = {
            key: sum(
                r['controls'].get(key, False) for r in {r['result']['item_id']: r for r in combined}.values()
            )
            for key in controls
        }
        if not all(aggregate_controls.values()):
            raise AssertionError('required control has no saved evidence')
        results = [ItemResult(**r['result']) for r in saved_rows]
        report = {
            'protocol_hash': phash,
            'split': SPLIT,
            'alpha': args.alpha,
            'controls': aggregate_controls,
            'controls_this_session': controls,
            'summary': summarise(results),
            'format_diagnostic': diagnose_format(results, agreement),
            'obfuscation': obfuscation_summary(results),
            'diagnostics': diagnostics_summary(results),
            'claim': 'MMLU utility measurement only. Item independence and independent '
            'subject blocks are not established; bounds are not a population '
            'guarantee. No free-generation or privacy claim.',
        }
        write_json(out_dir / 'report.json', report)
        common.event('completed', protocol_hash=phash, n_items=len(results))
    finally:
        probe.close()


if __name__ == '__main__':
    main()
