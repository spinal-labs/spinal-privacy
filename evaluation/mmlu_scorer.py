"""MMLU option-likelihood scorer, implementing the scoring protocol in PROTOCOL.md.

PROBLEM: generation with exact-match grading cannot score MMLU option likelihoods.

WHY: positional token agreement against the clean model's output measures whether the same
words came out in the same slots. That is not whether the answer is right. A protected
prompt can score low on token agreement while clean and protected outputs reach the same
conclusion.

METRIC: the regression rate -- of the questions the clean model gets right, the share the
protected model gets wrong. Improvements are counted separately and never offset them.

Every choice here is dictated by the frozen protocol. Where this file must choose something
the protocol left open, it raises rather than guessing.

Controls, all required before any result is interpreted:
  C1   identity transform reproduces the clean prediction PER ITEM (|dlogit| <= 1e-9)
  C1a  an independent oracle re-scores fixtures through ordinary input_ids
  C1b  constructed fixtures with known scores, ties, and every transition category
  C1c  END-TO-END: assert at the model-call boundary that the tensor the model receives is
       the transformed one. C1/C1a/C1b all pass a dispatch bug that sends clean embeddings.
"""

import hashlib
import inspect
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch

LABELS = (' A', ' B', ' C', ' D')
IDENTITY_TOL = 1e-9

# Diagnostic thresholds. Engineering defaults, not empirically derived.
MASS_FLOOR = 0.01  # per-item candidate mass below this flags the item
MASS_FLAG_SHARE = 0.05  # share of flagged items that flags the whole run
COLLAPSE_RATIO = 0.1  # protected mass below this multiple of clean mass = collapse
COLLAPSE_FLAG_SHARE = 0.05
FREEGEN_AGREEMENT = 0.95  # below this, format validation FAILS and the run stops


@dataclass
class ItemResult:
    """One scored item: clean and protected predictions, their scores, and per-row diagnostics."""

    item_id: str
    subject: str
    gold: int
    clean_pred: int
    protected_pred: int
    clean_scores: list[float]
    protected_scores: list[float]
    clean_correct: bool
    protected_correct: bool
    transition: str
    clean_mass: float
    protected_mass: float
    clean_margin: float
    protected_margin: float
    tie_clean: bool
    tie_protected: bool
    tokens_changed: int
    tokens_changed_bf16: int
    content_tokens: int
    prompt_tokens: int
    label_changed: bool
    # Diagnostic fields. None of them enters a scored quantity; they default to None so
    # rows without diagnostics still load in post-hoc tools. A row carries either all five
    # or none (validate()).
    clean_winner_protected_lead: float | None = None
    """Signed lead of the CLEAN winner in the protected arm: its protected score minus the
    best other protected score. Negative means the clean winner lost. protected_margin is the
    lead of whoever wins in the protected arm and does not say how far the clean winner moved."""
    clean_winner_lead_change: float | None = None
    """clean_winner_protected_lead minus the clean lead of the same option (clean_margin)."""
    displacement_mean: float | None = None
    """Mean over content tokens of ||z_t - e_t|| / ||e_t|| (fp32, delivered blend)."""
    lookup_margins: list[float] | None = None
    """Per content token, fp32: cosine of z_t with its own embedding minus the best cosine
    with any other vocabulary row. Negative means another token wins the nearest-neighbour
    lookup."""
    lookup_margins_bf16: list[float] | None = None
    """The same after casting z_t to bf16 (the delivered representation)."""


DIAGNOSTIC_FIELDS = (
    'clean_winner_protected_lead',
    'clean_winner_lead_change',
    'displacement_mean',
    'lookup_margins',
    'lookup_margins_bf16',
)
LOOKUP_MARGIN_TOL = 1e-5  # margins are recomputed with the same fp32 nearest-neighbour decoder


def clean_winner_lead(
    clean_scores: Sequence[float], protected_scores: Sequence[float]
) -> tuple[float, float]:
    """(signed protected lead of the clean winner, its change from the clean lead)."""
    winner, _, clean_lead = predict(list(clean_scores))
    if len(protected_scores) != 4 or not all(math.isfinite(v) for v in protected_scores):
        raise ValueError(f'expected 4 finite protected scores, got {protected_scores}')
    others = max(v for i, v in enumerate(protected_scores) if i != winner)
    lead = protected_scores[winner] - others
    return lead, lead - clean_lead


# Retain the actual decorator-generated functions, not just their names. A later
# hand-written __init__/__eq__ override must be fingerprinted (or rejected).
_DATACLASS_GENERATED = {
    name: fn
    for name, fn in vars(ItemResult).items()
    if name in {'__init__', '__repr__', '__eq__', '__hash__'}
    and callable(fn)
    and getattr(getattr(inspect.unwrap(fn), '__code__', None), 'co_filename', None) == '<string>'
}


def render(
    question: str, choices: Sequence[str], instruction: str | None = None, answer_line: bool = True
) -> str:
    """The user message; the chat template adds the assistant prefix.

    Defaults: question, four options, then `Answer:` inside the user message.
    Alternative: an optional instruction line replaces `Answer:` in the user message and
    `Answer:` becomes an assistant prefill appended by the runner after the chat template.
    The caller records which it used; both parameters enter the protocol hash.
    """
    if len(choices) != 4:
        raise ValueError(f'expected 4 choices, got {len(choices)}')
    if instruction is not None and (
        not isinstance(instruction, str) or not instruction.strip() or instruction != instruction.strip()
    ):
        raise ValueError('instruction must be a non-empty string without surrounding whitespace')
    body = '\n'.join(f'{letter}. {choice}' for letter, choice in zip('ABCD', choices))
    lines = [question, body]
    if instruction is not None:
        lines.append(instruction)
    if answer_line:
        lines.append('Answer:')
    return '\n'.join(lines)


def label_token_ids(tok: Any) -> tuple[list[list[int]], int]:
    """Encode the four candidate continuations and assert the protocol's equal-length rule.

    Equal characters does not imply equal tokens. If the counts differ,
    the justification for skipping length normalisation fails and the run must stop.
    """
    ids = [tok(label, add_special_tokens=False)['input_ids'] for label in LABELS]
    lengths = {len(x) for x in ids}
    if len(lengths) != 1:
        raise ValueError(
            f'candidate continuations tokenize to unequal lengths {lengths}; the protocol forbids proceeding'
        )
    length = lengths.pop()
    if (
        length == 0
        or len({tuple(x) for x in ids}) != 4
        or not all(type(i) is int and i >= 0 for candidate in ids for i in candidate)
    ):
        raise ValueError('labels must encode to distinct nonempty non-negative integer sequences')
    return ids, length


def protocol_hash() -> str:
    """Identity of the measurement contract, for the resume key and the manifest."""
    # Every function whose behaviour changes a number, plus the row schema. Omitting
    # clopper_pearson_upper would let a replacement that returns zero leave the hash
    # UNCHANGED.
    logic = ''.join(
        inspect.getsource(f)
        for f in (
            render,
            label_token_ids,
            score_options,
            predict,
            transition,
            validate,
            summarise,
            diagnose_format,
            clopper_pearson_upper,
            _binom_tail_log,
            obfuscation_summary,
            clean_winner_lead,
            _validate_diagnostics,
            diagnostics_summary,
        )
    )
    # Source text alone is not a behaviour fingerprint. It misses an
    # ItemResult.__post_init__ that silently overwrites protected_mass, and a mutation of
    # clopper_pearson_upper.__defaults__ that changes alpha from 0.05 to 0.5. So: hash the
    # defaults, the presence of any dataclass hook, and a behavioural probe of the bound.
    # Hook PRESENCE is not hook behaviour: two __post_init__ hooks that overwrite
    # protected_mass with 0.0 versus 0.5 would hash identically. Hash the source of each
    # hand-written hook, rejecting unreadable source.
    # Only hand-written hooks are hashed. The dataclass decorator generates __eq__ and
    # __repr__ dynamically, so inspect.getsource raises OSError on them, and a repr()
    # fallback would embed a memory address and make the hash differ between processes
    # (every run non-reproducible, resume rejecting its own rows). Only actual generated
    # members are excluded; an unreadable hand-written hook is a hard error rather than a
    # silent address.
    hooks = {}
    for h, fn in sorted(vars(ItemResult).items()):
        if not callable(fn) or not h.startswith('__') or fn is _DATACLASS_GENERATED.get(h):
            continue
        try:
            hooks[h] = inspect.getsource(fn)
        except (OSError, TypeError) as exc:
            raise RuntimeError(
                f'cannot read the source of ItemResult.{h}; refusing to hash an unstable fallback ({exc})'
            ) from None
    probe = [clopper_pearson_upper(0, 299), clopper_pearson_upper(4, 1000), clopper_pearson_upper(119, 13866)]
    protected = {
        'logic_sha': hashlib.sha256(logic.encode()).hexdigest()[:16],
        'schema': [f.name for f in fields(ItemResult)],
        'class_source': inspect.getsource(ItemResult),
        'dataclass_hooks': hooks,
        'defaults': {
            f.__name__: repr(f.__defaults__)
            for f in (render, score_options, predict, clopper_pearson_upper, diagnose_format, protocol_hash)
        },
        'bound_probe': [round(v, 12) for v in probe],
        'labels': LABELS,
        'identity_tol': IDENTITY_TOL,
        'mass_floor': MASS_FLOOR,
        'mass_flag_share': MASS_FLAG_SHARE,
        'collapse_ratio': COLLAPSE_RATIO,
        'collapse_flag_share': COLLAPSE_FLAG_SHARE,
        'freegen_agreement': FREEGEN_AGREEMENT,
        'revision': 5,
        'diagnostic_fields': list(DIAGNOSTIC_FIELDS),
        'lookup_margin_tol': LOOKUP_MARGIN_TOL,
        'amendments': ['diagnostic fields'],
    }
    return hashlib.sha256(json.dumps(protected, sort_keys=True).encode()).hexdigest()[:16]


@torch.no_grad()
def score_options(
    model: Any, embeds: torch.Tensor | None, label_ids: list[list[int]]
) -> tuple[list[float], list[float], float]:
    """Return (per-option score, per-option log-probability, total candidate mass).

    Ranking rules:
      - SINGLE-token candidates are ranked on RAW LOGITS. They share a prefix, so the
        log-normaliser is common to all four and never needs computing. Ranking on separately
        rounded log-probabilities can tie two distinct logits and flip the answer; in fp32
        the logits {0, 2^-27, -1, -2, 10} show this.
      - Log-probabilities are still computed, in fp64, but only for the mass diagnostic.
      - MULTI-token candidates genuinely have differing normalisers; those are summed in fp64.
    """
    if len(label_ids) != 4:
        raise ValueError(f'expected exactly 4 candidate continuations, got {len(label_ids)}')
    flat = [i for x in label_ids for i in x]
    if not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in flat):
        raise ValueError(
            f'continuation token ids must be non-negative ints, got {label_ids}; '
            'a negative id silently selects from the end of the vocabulary'
        )
    if len({tuple(x) for x in label_ids}) != 4:
        raise ValueError(f'candidate continuations must be distinct, got {label_ids}')
    lengths = {len(x) for x in label_ids}
    if len(lengths) != 1:
        raise ValueError(f'candidate continuations have unequal token lengths {lengths}')
    length = lengths.pop()
    if length == 0:
        raise ValueError('empty continuation')
    single_token = length == 1  # DERIVED, never taken on trust
    if single_token:
        out = model(inputs_embeds=embeds, use_cache=False)
        if out.logits.ndim != 3 or out.logits.shape[0] != 1 or out.logits.shape[1] == 0:
            raise ValueError('expected nonempty batch-size-one logits')
        last = out.logits[:, -1].double()
        idx = torch.tensor([x[0] for x in label_ids], device=last.device)
        scores = last[0, idx]  # RAW logits: the ranking quantity
        logprobs = torch.log_softmax(last[0], -1)[idx]  # diagnostics only
        if not torch.isfinite(scores).all() or not torch.isfinite(logprobs).all():
            raise ValueError('non-finite option scores; the protocol requires aborting')
        return scores.tolist(), logprobs.tolist(), float(logprobs.exp().sum())
    # Full, uncached teacher forcing per branch: no shared mutable KV cache. Only
    # clean continuation embeddings are appended to the already transformed prompt.
    if embeds is None or embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[1] == 0:
        raise ValueError('multi-token scoring needs one nonempty embedded prompt')
    joint = []
    for candidate in label_ids:
        tail = torch.tensor([candidate[:-1]], device=embeds.device)
        suffix = model.get_input_embeddings()(tail).to(embeds.dtype)
        branch = torch.cat((embeds, suffix), dim=1)
        logits = model(inputs_embeds=branch, use_cache=False).logits
        if logits.shape[:2] != branch.shape[:2]:
            raise ValueError('model logits do not cover the complete branch')
        positions = logits[0, embeds.shape[1] - 1 : embeds.shape[1] + length - 1].double()
        targets = torch.tensor(candidate, device=positions.device)
        lp = positions.log_softmax(-1).gather(1, targets[:, None]).sum()
        if not torch.isfinite(lp):
            raise ValueError('non-finite multi-token score')
        joint.append(float(lp))
    return joint, list(joint), sum(math.exp(x) for x in joint)


def predict(scores: Sequence[float]) -> tuple[int, bool, float]:
    """Lowest-index tie-break, per the protocol, with the tie reported not hidden."""
    if len(scores) != 4 or not all(math.isfinite(v) for v in scores):
        raise ValueError(f'expected 4 finite option scores, got {scores}')
    best = max(scores)
    winners = [i for i, v in enumerate(scores) if v == best]
    rest = sorted((v for i, v in enumerate(scores) if i != winners[0]), reverse=True)
    return winners[0], len(winners) > 1, best - rest[0]


def transition(clean_ok: bool, prot_ok: bool) -> str:
    """Name the clean-to-protected correctness transition."""
    return {
        (True, True): 'right_right',
        (True, False): 'right_wrong',
        (False, True): 'wrong_right',
        (False, False): 'wrong_wrong',
    }[(clean_ok, prot_ok)]


def validate(results: list[ItemResult]) -> list[ItemResult]:
    """Reject rows that contradict themselves.

    Trusting supplied booleans and transition strings would let a row with
    clean_correct=True, protected_correct=False and transition='right_right' report zero
    regressions and retention 1.0, contradicting its own inputs. Correctness and transition
    are DERIVED here from prediction versus gold, so a caller cannot assert them.
    """
    seen = set()
    for r in results:
        if r.item_id in seen:
            raise ValueError(
                f'duplicate item_id {r.item_id!r}; duplicated rows inflate the '
                'denominator and shrink the bound'
            )
        seen.add(r.item_id)
        for name, v in (('item_id', r.item_id), ('subject', r.subject)):
            if not isinstance(v, str) or not v.strip():
                raise ValueError(
                    f'row has {name}={v!r}; a missing identifier makes the row '
                    'untraceable and duplicate detection meaningless'
                )
        for name, v in (
            ('clean_correct', r.clean_correct),
            ('protected_correct', r.protected_correct),
            ('tie_clean', r.tie_clean),
            ('tie_protected', r.tie_protected),
            ('label_changed', r.label_changed),
        ):
            if not isinstance(v, bool):
                raise ValueError(
                    f'{r.item_id}: {name}={v!r} must be a bool; 1.0 passed the '
                    'equality checks and then crashed the confidence bound'
                )
        for name, v in (('clean_margin', r.clean_margin), ('protected_margin', r.protected_margin)):
            if isinstance(v, bool) or not isinstance(v, float) or v < 0.0:
                raise ValueError(f'{r.item_id}: {name}={v!r} must be a non-negative float')
        for name, v in (('gold', r.gold), ('clean_pred', r.clean_pred), ('protected_pred', r.protected_pred)):
            if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 3:
                raise ValueError(f'{r.item_id}: {name}={v!r} must be an int in 0..3')
        # THE PREDICTION MUST FOLLOW FROM THE SCORES. Without this, a row whose scores
        # clearly select option 1 could declare protected_pred=0 and report zero regressions
        # at retention 1.0 -- a real regression rendered invisible.
        for arm, scores, pred, tie, margin in (
            ('clean', r.clean_scores, r.clean_pred, r.tie_clean, r.clean_margin),
            ('protected', r.protected_scores, r.protected_pred, r.tie_protected, r.protected_margin),
        ):
            exp_pred, exp_tie, exp_margin = predict(list(scores))
            if pred != exp_pred:
                raise ValueError(
                    f'{r.item_id}: {arm}_pred={pred} contradicts its scores '
                    f'{list(scores)}, which select {exp_pred}'
                )
            if tie != exp_tie:
                raise ValueError(f'{r.item_id}: {arm} tie flag {tie} contradicts its scores')
            if abs(margin - exp_margin) > 1e-12:
                raise ValueError(
                    f'{r.item_id}: {arm}_margin {margin} != {exp_margin} computed from the scores'
                )
        for name, v in (('clean_mass', r.clean_mass), ('protected_mass', r.protected_mass)):
            # ASYMMETRIC tolerance. Summing four fp64 probabilities that cover the whole
            # distribution legitimately returns 1.0000000000000002, so rounding ABOVE one is
            # tolerated. A sum of exponentials cannot go negative through rounding, so the
            # lower bound stays exactly zero; a symmetric tolerance would accept -5e-10,
            # which no correct computation produces.
            if isinstance(v, bool) or not isinstance(v, float) or not 0.0 <= v <= 1.0 + 1e-9:
                raise ValueError(
                    f'{r.item_id}: {name}={v!r} must be a float in '
                    '[0, 1+1e-9]; bool and negative are both rejected'
                )
        for name, v in (
            ('tokens_changed', r.tokens_changed),
            ('tokens_changed_bf16', r.tokens_changed_bf16),
            ('content_tokens', r.content_tokens),
            ('prompt_tokens', r.prompt_tokens),
        ):
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(
                    f'{r.item_id}: {name}={v!r} must be a non-negative int (bool is not an int here)'
                )
        if r.tokens_changed > r.content_tokens or r.tokens_changed_bf16 > r.content_tokens:
            raise ValueError(f'{r.item_id}: changed tokens exceed content tokens')
        if r.content_tokens > r.prompt_tokens:
            raise ValueError(
                f'{r.item_id}: content_tokens {r.content_tokens} exceeds prompt_tokens {r.prompt_tokens}'
            )
        for name, v in (
            ('clean_mass', r.clean_mass),
            ('protected_mass', r.protected_mass),
            ('clean_margin', r.clean_margin),
            ('protected_margin', r.protected_margin),
        ):
            if not math.isfinite(v):
                raise ValueError(
                    f'{r.item_id}: non-finite {name}={v}; the protocol requires '
                    'aborting, never silently dropping'
                )
        for name, vec in (('clean_scores', r.clean_scores), ('protected_scores', r.protected_scores)):
            if len(vec) != 4 or not all(math.isfinite(v) for v in vec):
                raise ValueError(f'{r.item_id}: {name} must be 4 finite values, got {vec}')
        if r.clean_correct != (r.clean_pred == r.gold):
            raise ValueError(f'{r.item_id}: clean_correct contradicts clean_pred vs gold')
        if r.protected_correct != (r.protected_pred == r.gold):
            raise ValueError(f'{r.item_id}: protected_correct contradicts protected_pred vs gold')
        if r.transition != transition(r.clean_correct, r.protected_correct):
            raise ValueError(f'{r.item_id}: transition {r.transition!r} contradicts the correctness flags')
        if r.label_changed != (r.clean_pred != r.protected_pred):
            raise ValueError(f'{r.item_id}: label_changed contradicts the predictions')
        _validate_diagnostics(r)
    return results


def _validate_diagnostics(r: ItemResult) -> None:
    """All five diagnostic fields or none, each consistent."""
    present = [getattr(r, f) is not None for f in DIAGNOSTIC_FIELDS]
    if not any(present):
        return
    if not all(present):
        raise ValueError(
            f'{r.item_id}: diagnostic fields are partial; a row carries all of {DIAGNOSTIC_FIELDS} or none'
        )
    for name in ('clean_winner_protected_lead', 'clean_winner_lead_change', 'displacement_mean'):
        v = getattr(r, name)
        if isinstance(v, bool) or not isinstance(v, float) or not math.isfinite(v):
            raise ValueError(f'{r.item_id}: {name}={v!r} must be a finite float')
    if r.displacement_mean < 0.0:
        raise ValueError(f'{r.item_id}: displacement_mean must be non-negative')
    lead, change = clean_winner_lead(list(r.clean_scores), list(r.protected_scores))
    if abs(lead - r.clean_winner_protected_lead) > 1e-12 or abs(change - r.clean_winner_lead_change) > 1e-12:
        raise ValueError(
            f'{r.item_id}: clean-winner lead {r.clean_winner_protected_lead}/'
            f'{r.clean_winner_lead_change} disagree with the scores ({lead}, {change})'
        )
    for name, changed in (
        ('lookup_margins', r.tokens_changed),
        ('lookup_margins_bf16', r.tokens_changed_bf16),
    ):
        margins = getattr(r, name)
        if (
            not isinstance(margins, list)
            or len(margins) != r.content_tokens
            or not all(
                isinstance(v, float) and not isinstance(v, bool) and math.isfinite(v) and -2.0 <= v <= 2.0
                for v in margins
            )
        ):
            raise ValueError(
                f'{r.item_id}: {name} must hold one finite float in [-2, 2] per '
                f'content token ({r.content_tokens})'
            )
        # A strictly negative margin means another row wins the lookup; a zero margin is a
        # tie the decoder breaks to the lower id. So negatives <= changed <= non-positives,
        # up to the tolerance between two fp32 evaluations of the same cosine.
        negative = sum(v < -LOOKUP_MARGIN_TOL for v in margins)
        non_positive = sum(v <= LOOKUP_MARGIN_TOL for v in margins)
        if not negative <= changed <= non_positive:
            raise ValueError(
                f'{r.item_id}: {name} has {negative} negative and {non_positive} '
                f'non-positive margins but {changed} changed tokens'
            )


def summarise(results: list[ItemResult]) -> dict:
    """Regression rate; improvements separate; retention as context."""
    validate(results)
    n = len(results)
    clean_correct = sum(r.clean_correct for r in results)
    x = sum(r.transition == 'right_wrong' for r in results)
    y = sum(r.transition == 'wrong_right' for r in results)
    r_rate = x / clean_correct if clean_correct else None
    subjects = {}
    for subject in sorted({r.subject for r in results}):
        rows = [r for r in results if r.subject == subject]
        nc = sum(r.clean_correct for r in rows)
        xr = sum(r.transition == 'right_wrong' for r in rows)
        subjects[subject] = {
            'n_items': len(rows),
            'clean_correct': nc,
            'protected_correct': sum(r.protected_correct for r in rows),
            'regressions': xr,
            'improvements': sum(r.transition == 'wrong_right' for r in rows),
            'regression_rate': xr / nc if nc else None,
            'clean_accuracy': nc / len(rows),
            'protected_accuracy': sum(r.protected_correct for r in rows) / len(rows),
            'obfuscation': obfuscation_summary(rows),
        }
    weights2 = (
        sum((v['clean_correct'] / clean_correct) ** 2 for v in subjects.values()) if clean_correct else None
    )
    allowance = math.sqrt(math.log(20) * weights2 / 2) if weights2 else None
    return {
        'n_items': n,
        'clean_accuracy': clean_correct / n if n else None,
        'protected_accuracy': sum(r.protected_correct for r in results) / n if n else None,
        'per_subject': subjects,
        'subject_block_upper_95': min(1.0, r_rate + allowance) if allowance else None,
        'subject_block_allowance': allowance,
        'effective_subject_blocks': 1 / weights2 if weights2 else None,
        'subject_block_assumption': 'Independent subject blocks conditional on the clean '
        'outcomes; weights are clean-correct counts, not total subject sizes. '
        'Independence is not established; no population guarantee.',
        'regression_lower_95': (
            0.0 if x == 0 else 1 - clopper_pearson_upper(clean_correct - x, clean_correct)
        )
        if clean_correct
        else None,
        'predicted_label_counts': {
            arm: [sum(getattr(r, arm + '_pred') == i for r in results) for i in range(4)]
            for arm in ('clean', 'protected')
        },
        'prompt_tokens': {
            'min': min((r.prompt_tokens for r in results), default=None),
            'max': max((r.prompt_tokens for r in results), default=None),
            'mean': sum(r.prompt_tokens for r in results) / n if n else None,
        },
        'clean_correct': clean_correct,
        'protected_correct': sum(r.protected_correct for r in results),
        'regressions': x,
        'improvements': y,
        'regression_rate': r_rate,
        'regression_upper_95': clopper_pearson_upper(x, clean_correct) if clean_correct else None,
        'retention': (1 - r_rate + y / clean_correct) if clean_correct else None,
        'label_change_rate': sum(r.label_changed for r in results) / n if n else None,
        'transitions': {
            k: sum(r.transition == k for r in results)
            for k in ('right_right', 'right_wrong', 'wrong_right', 'wrong_wrong')
        },
        'ties_clean': sum(r.tie_clean for r in results),
        'ties_protected': sum(r.tie_protected for r in results),
        'note': 'retention is context, never the gate. Improvements never offset regressions. '
        'The Clopper-Pearson bound assumes independent items; MMLU clusters within '
        'subjects, so quote it with that assumption stated.',
    }


def obfuscation_summary(results: list[ItemResult]) -> dict:
    """Both estimands and precisions, with empty-content prompts excluded from macro."""
    rows = [r for r in results if r.content_tokens]
    total = sum(r.content_tokens for r in rows)
    out = {'zero_content_items': len(results) - len(rows)}
    for suffix, field in (('', 'tokens_changed'), ('_bf16', 'tokens_changed_bf16')):
        out['macro_pct' + suffix] = (
            sum(100 * getattr(r, field) / r.content_tokens for r in rows) / len(rows) if rows else None
        )
        out['pooled_pct' + suffix] = 100 * sum(getattr(r, field) for r in rows) / total if total else None
    return out


def diagnostics_summary(results: list[ItemResult]) -> dict:
    """Descriptive only; nothing here enters a gate or a bound."""
    rows = [r for r in results if r.clean_winner_protected_lead is not None]
    out = {'n_rows_with_diagnostics': len(rows), 'n_rows': len(results)}
    if not rows:
        return out

    def stats(values: list[float]) -> dict | None:
        v = sorted(values)
        if not v:
            return None
        return {
            'n': len(v),
            'mean': sum(v) / len(v),
            'min': v[0],
            'p10': v[int(0.1 * len(v))],
            'median': v[len(v) // 2],
            'max': v[-1],
        }

    out['displacement_mean'] = stats([r.displacement_mean for r in rows])
    by_transition = {}
    for key in ('right_right', 'right_wrong', 'wrong_right', 'wrong_wrong'):
        group = [r for r in rows if r.transition == key]
        by_transition[key] = {
            'clean_winner_protected_lead': stats([r.clean_winner_protected_lead for r in group]),
            'clean_winner_lead_change': stats([r.clean_winner_lead_change for r in group]),
            'clean_winner_lost_lead': sum(r.clean_winner_protected_lead < 0 for r in group),
        }
    out['by_transition'] = by_transition
    for suffix, field in (('', 'lookup_margins'), ('_bf16', 'lookup_margins_bf16')):
        pooled = [v for r in rows for v in getattr(r, field)]
        out['lookup_margin' + suffix] = {
            **(stats(pooled) or {}),
            'share_negative': sum(v < 0 for v in pooled) / len(pooled) if pooled else None,
            'share_below_0.01': sum(v < 0.01 for v in pooled) / len(pooled) if pooled else None,
        }
    out['note'] = (
        'Descriptive diagnostics (PROTOCOL.md): the '
        'clean-winner lead separates how far the clean choice moved from who wins; '
        'displacement and lookup margins separate large input movement hidden by the '
        'decoder from boundary crossings. No gate, no bound, no claim.'
    )
    return out


def _binom_tail_log(x: int, n: int, p: float) -> float:
    """P(X <= x) for X ~ Bin(n, p), summed in LOG space.

    The direct form math.comb(n,k) * p**k * (1-p)**(n-k) overflows: comb(10000, 200) cannot
    be converted to a float even though the probability is perfectly representable, so
    clopper_pearson_upper(200, 10000) would raise OverflowError.
    """
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, l1p = math.log(p), math.log1p(-p)
    terms = [
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * lp + (n - k) * l1p
        for k in range(x + 1)
    ]
    top = max(terms)
    return math.exp(top + math.log(sum(math.exp(t - top) for t in terms)))


def clopper_pearson_upper(x: int, n: int, alpha: float = 0.05) -> float | None:
    """Exact one-sided upper bound on a binomial rate.

    With x=0 this is 1 - alpha**(1/n), which is also why an all-zero bootstrap is wrong:
    resampling zeros can only ever produce zero.
    """
    if not isinstance(x, int) or not isinstance(n, int) or isinstance(x, bool) or isinstance(n, bool):
        raise TypeError(f'x and n must be ints, got {type(x).__name__}, {type(n).__name__}')
    if not isinstance(alpha, float | int) or not 0.0 < alpha < 1.0 or alpha != alpha:
        raise ValueError(
            f'alpha must lie strictly in (0, 1); got {alpha!r}. Unchecked values '
            'can return negative, NaN or complex bounds.'
        )
    if n < 0 or x < 0:
        raise ValueError(f'negative counts: x={x}, n={n}')
    if x > n:
        raise ValueError(f'x={x} exceeds n={n}; invalid data, not a bound to compute')
    if n == 0:
        return None
    if x == 0:
        return 1 - alpha ** (1 / n)
    if x == n:
        return 1.0
    # Bracket from 0, not x/n: for alpha above the observed rate the solution lies BELOW
    # x/n, and a bracket starting at x/n would exclude it (cp(1, 10, alpha=0.9) is ~0.0545,
    # not 0.1). The protocol only uses alpha=0.05, but other alpha values stay correct.
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _binom_tail_log(x, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def diagnose_format(results: list[ItemResult], freegen_agreement: dict[str, float] | None = None) -> dict:
    """Runs BEFORE any result is interpreted.

    Catches an instruction-tuned model that will not emit a bare label -- the scorer would
    then rank four near-zero strings and still look confident -- and the subtler case where
    protection collapses the absolute mass while leaving the ranking intact.
    """
    n = len(results)
    if not n:
        raise ValueError('diagnose_format called with no items; an empty run cannot be declared ok')
    validate(results)  # guards against a row with mass 2.0 reaching status 'ok'
    # NaN comparisons are all False, so an all-NaN run would otherwise report status 'ok'.
    bad = [
        r.item_id for r in results if not (math.isfinite(r.clean_mass) and math.isfinite(r.protected_mass))
    ]
    if bad:
        raise ValueError(
            f'non-finite candidate mass on {len(bad)} items (e.g. {bad[:3]}); the protocol requires aborting'
        )
    low_clean = sum(r.clean_mass < MASS_FLOOR for r in results)
    low_prot = sum(r.protected_mass < MASS_FLOOR for r in results)
    collapsed = sum(r.protected_mass < COLLAPSE_RATIO * r.clean_mass for r in results)
    flags = []
    if low_clean / n >= MASS_FLAG_SHARE:
        flags.append(f'clean candidate mass below {MASS_FLOOR} on {low_clean}/{n} items')
    if low_prot / n >= MASS_FLAG_SHARE:
        flags.append(f'protected candidate mass below {MASS_FLOOR} on {low_prot}/{n} items')
    if collapsed / n >= COLLAPSE_FLAG_SHARE:
        flags.append(f'protected mass collapsed below {COLLAPSE_RATIO}x clean on {collapsed}/{n} items')
    # Below FREEGEN_AGREEMENT, format validation FAILS and the run stops.
    if freegen_agreement is None:
        flags.append(
            'free-generation agreement NOT measured; the protocol requires it '
            'before any preservation or acceptance claim'
        )
    else:
        # Must be PER ARM. A single pooled scalar can hide an arm: clean 285/285 with
        # protected 257/285 pools to 95.09% and passes, while protected agreement is only
        # 90.18%.
        if isinstance(freegen_agreement, (int, float)):
            raise TypeError(
                'freegen_agreement must be a mapping of arm -> value, not a '
                'single pooled scalar; pooling can hide a failing arm'
            )
        for arm in ('clean', 'protected'):
            if arm not in freegen_agreement:
                raise ValueError(f'freegen_agreement missing the {arm!r} arm')
        for arm, v in freegen_agreement.items():
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or not 0.0 <= v <= 1.0
            ):
                raise ValueError(
                    f'freegen_agreement[{arm!r}]={v!r} is not a finite fraction '
                    'in [0, 1]; NaN, inf and 1.5 are rejected'
                )
            if v < FREEGEN_AGREEMENT:
                raise ValueError(
                    f'free-generation agreement for {arm} is {v:.4f}, below '
                    f'{FREEGEN_AGREEMENT}; format validation failed, stop'
                )
    return {
        'n': n,
        'low_mass_clean': low_clean,
        'low_mass_protected': low_prot,
        'mass_collapsed': collapsed,
        'flags': flags,
        'freegen_agreement': freegen_agreement,
        'mean_mass_clean': sum(r.clean_mass for r in results) / n,
        'mean_mass_protected': sum(r.protected_mass for r in results) / n,
        'status': 'flagged' if flags else 'ok',
    }
