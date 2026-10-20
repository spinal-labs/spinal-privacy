"""C1b: constructed fixtures for the MMLU scorer. No GPU, no model.

See PROTOCOL.md. C1 is near-tautological on its own, because bitwise-identical inputs down
a shared path almost have to agree. C1b exercises the arithmetic and bookkeeping with known
answers, so a bug that BOTH paths share -- a reversed label map, a shifted likelihood -- is
caught rather than reproduced identically on each side.
"""

import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.mmlu_scorer import (  # noqa: E402
    COLLAPSE_RATIO,
    MASS_FLOOR,
    ItemResult,
    clopper_pearson_upper,
    diagnose_format,
    label_token_ids,
    predict,
    render,
    score_options,
    summarise,
    transition,
    validate,
)

# Labels are deliberately NOT vocabulary ids 0..3. A scorer that ignores label_ids and
# reads the first four vocabulary slots must FAIL these fixtures.
LABEL_IDS = [[41], [17], [93], [64]]


class EchoModel:
    """Logits derived from the last embedding position, so discarding the input changes the answer."""

    def __init__(self, vocab: int = 120) -> None:
        self.vocab = vocab

    def __call__(self, inputs_embeds: torch.Tensor | None = None, use_cache: bool = False) -> SimpleNamespace:
        if inputs_embeds is None:
            raise AssertionError('scorer must supply the embeddings')
        v = torch.full((self.vocab,), -50.0, dtype=torch.float32)
        for (i,), off in zip(LABEL_IDS, range(4)):
            v[i] = float(inputs_embeds[0, -1, off])
        return SimpleNamespace(logits=v.reshape(1, 1, -1))


class FakeModel:
    """Multi-position logits, so reading the WRONG position is detectable.

    With shape (1, 1, vocab) the first and last position would be identical, and a scorer
    reading position 0 would pass. Here earlier positions carry different values, so only
    the final position yields the intended scores.
    """

    def __init__(self, final_logits: list[float], n_positions: int = 3) -> None:
        vocab = len(final_logits)
        rows = [[float(-1000 - i) for _ in range(vocab)] for i in range(n_positions - 1)]
        rows.append(list(final_logits))
        self._logits = torch.tensor(rows, dtype=torch.float32)

    def __call__(self, inputs_embeds: torch.Tensor | None = None, use_cache: bool = False) -> SimpleNamespace:
        return SimpleNamespace(logits=self._logits.unsqueeze(0))


def logits_with(vals: list[float], ids: list[list[int]] = LABEL_IDS, vocab: int = 120) -> list[float]:
    """Place known values at the label ids; everything else is a decoy."""
    v = [-50.0] * vocab
    for (i,), x in zip(ids, vals):
        v[i] = x
    return v


def item(
    gold: int = 0, clean_pred: int | None = None, protected_pred: int | None = None, **kw: Any
) -> ItemResult:
    """Build a SELF-CONSISTENT row.

    Correctness, transition and label_changed are DERIVED from the predictions, never passed
    in, so a fixture cannot assert them independently of the predictions. validate() rejects
    rows where they disagree.
    """
    clean_pred = gold if clean_pred is None else clean_pred
    protected_pred = clean_pred if protected_pred is None else protected_pred
    ok_c, ok_p = clean_pred == gold, protected_pred == gold

    # Scores are DERIVED from the predictions. With equal-score defaults a supplied
    # prediction could contradict its scores and still validate.
    def mk(p: int) -> list[float]:
        return [2.0 if i == p else 1.0 for i in range(4)]

    base = dict(
        item_id='x',
        subject='s',
        gold=gold,
        clean_pred=clean_pred,
        protected_pred=protected_pred,
        clean_scores=mk(clean_pred),
        protected_scores=mk(protected_pred),
        clean_correct=ok_c,
        protected_correct=ok_p,
        transition=transition(ok_c, ok_p),
        clean_mass=0.9,
        protected_mass=0.9,
        clean_margin=1.0,
        protected_margin=1.0,
        tie_clean=False,
        tie_protected=False,
        tokens_changed=0,
        tokens_changed_bf16=0,
        content_tokens=10,
        prompt_tokens=20,
        label_changed=clean_pred != protected_pred,
    )
    base.update(kw)
    return ItemResult(**base)


def test_render_is_exactly_the_protocol_format() -> None:
    out = render('Q?', ['a', 'b', 'c', 'd'])
    assert out == 'Q?\nA. a\nB. b\nC. c\nD. d\nAnswer:'
    # 'Answer:' sits INSIDE the user message; the chat template appends the assistant
    # prefix, and the continuation attaches after THAT. See PROTOCOL.md.
    assert out.endswith('Answer:')


def test_render_rejects_wrong_choice_count() -> None:
    try:
        render('Q?', ['a', 'b', 'c'])
    except ValueError:
        return
    raise AssertionError('render must reject a non-four choice list')


def test_single_token_ranking_uses_raw_logits_not_logprobs() -> None:
    """An fp32 counterexample (log-softmax ties two distinct logits), at non-trivial label ids."""
    v = logits_with([0.0, 2**-27, -1.0, -2.0])
    v[7] = 10.0  # unrelated mass, not a candidate
    scores, logprobs, mass = score_options(FakeModel(v), None, LABEL_IDS)
    assert scores[1] > scores[0], 'raw-logit ranking must separate the two candidates'
    lp32 = torch.log_softmax(torch.tensor(v), 0)
    assert lp32[LABEL_IDS[0][0]].item() == lp32[LABEL_IDS[1][0]].item(), 'fp32 must tie them'
    assert predict(scores)[0] == 1
    assert 0.0 < mass < 1.0


def test_scorer_reads_the_label_ids_not_vocabulary_slots_0_to_3() -> None:
    """A scorer ignoring label_ids and reading ids 0..3 must fail."""
    v = logits_with([1.0, 2.0, 3.0, 9.0])
    for i in range(4):
        v[i] = 100.0 + i  # junk at 0..3; highest at id 3
    scores, _, _ = score_options(FakeModel(v), None, LABEL_IDS)
    assert scores == [1.0, 2.0, 3.0, 9.0], f'read the wrong vocabulary slots: {scores}'
    assert predict(scores)[0] == 3


def test_scorer_reads_the_final_position() -> None:
    """Earlier positions hold large negative values; reading them yields the wrong scores."""
    v = logits_with([5.0, 1.0, 1.0, 1.0])
    scores, _, _ = score_options(FakeModel(v, n_positions=4), None, LABEL_IDS)
    assert scores == [5.0, 1.0, 1.0, 1.0], f'read a non-final position: {scores}'


def test_returned_logprobs_are_the_predetermined_values() -> None:
    """Returning zeros for the log-probability vector must fail."""
    v = logits_with([0.0, 0.0, 0.0, 0.0])
    _, logprobs, mass = score_options(FakeModel(v), None, LABEL_IDS)
    expect = torch.log_softmax(torch.tensor(v, dtype=torch.float64), 0)
    for got, i in zip(logprobs, [x[0] for x in LABEL_IDS]):
        assert abs(got - expect[i].item()) < 1e-9, 'log-probabilities are not the real ones'
    assert all(lp < 0 for lp in logprobs), 'zeros would mean probability 1 for every option'


def test_predict_breaks_ties_to_the_lowest_index_and_reports_them() -> None:
    idx, tie, margin = predict([2.0, 2.0, 1.0, 0.0])
    assert (idx, tie) == (0, True) and margin == 0.0
    idx, tie, margin = predict([1.0, 3.0, 1.0, 0.0])
    assert (idx, tie, margin) == (1, False, 2.0)


def test_predict_rejects_non_finite_and_wrong_length() -> None:
    for bad in ([float('nan'), 1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [float('inf')] * 4):
        try:
            predict(bad)
        except ValueError:
            continue
        raise AssertionError(f'predict must reject {bad}')


def test_multitoken_requires_real_prompt_embeddings() -> None:
    import pytest

    with pytest.raises(ValueError, match='nonempty embedded prompt'):
        score_options(
            FakeModel(logits_with([1.0, 2.0, 3.0, 4.0])), None, [[41, 5], [17, 5], [93, 5], [64, 5]]
        )


def test_score_options_rejects_unequal_label_lengths() -> None:
    try:
        score_options(FakeModel(logits_with([1.0, 2.0, 3.0, 4.0])), None, [[41], [17, 5], [93], [64]])
    except ValueError:
        return
    raise AssertionError('unequal continuation lengths must raise')


def test_mass_is_the_probability_of_the_four_candidates() -> None:
    v = logits_with([0.0, 0.0, 0.0, 0.0])
    v[7] = 100.0  # almost all mass on a non-candidate
    _, _, mass = score_options(FakeModel(v), None, LABEL_IDS)
    assert mass < MASS_FLOOR, f'expected collapsed candidate mass, got {mass}'


def test_transition_categories_are_exhaustive() -> None:
    assert transition(True, True) == 'right_right'
    assert transition(True, False) == 'right_wrong'
    assert transition(False, True) == 'wrong_right'
    assert transition(False, False) == 'wrong_wrong'


def test_regressions_and_improvements_never_offset() -> None:
    """Twenty broken and nineteen accidentally fixed must NOT read as almost no change."""
    rows = (
        [item(item_id=f'rw{i}', gold=0, clean_pred=0, protected_pred=1) for i in range(20)]
        + [item(item_id=f'wr{i}', gold=0, clean_pred=1, protected_pred=0) for i in range(19)]
        + [item(item_id=f'rr{i}', gold=0) for i in range(61)]
    )
    s = summarise(rows)
    assert s['regressions'] == 20 and s['improvements'] == 19
    assert s['clean_correct'] == 81
    assert abs(s['regression_rate'] - 20 / 81) < 1e-12
    # retention looks almost flat while 20 answers are broken -- the reason it is not the gate
    assert s['retention'] > 0.98


def test_clopper_pearson_zero_event_matches_the_closed_form() -> None:
    for n in (299, 492, 1000):
        assert abs(clopper_pearson_upper(0, n) - (1 - 0.05 ** (1 / n))) < 1e-12
    assert clopper_pearson_upper(0, 299) < 0.01  # the 299-item figure in the protocol
    assert clopper_pearson_upper(0, 298) > 0.01  # ...and it is tight
    assert clopper_pearson_upper(5, 100) > clopper_pearson_upper(0, 100)
    assert clopper_pearson_upper(100, 100) == 1.0


def test_clopper_pearson_bounds_the_binomial_tail() -> None:
    x, n = 3, 200
    u = clopper_pearson_upper(x, n)
    tail = sum(math.comb(n, k) * u**k * (1 - u) ** (n - k) for k in range(x + 1))
    assert abs(tail - 0.05) < 1e-6, f'upper bound must put 5% in the lower tail, got {tail}'


def test_format_diagnostic_catches_collapsed_mass_with_intact_ranking() -> None:
    """The failure the diagnostic exists for: predictions unchanged, mass gone."""
    rows = [
        item(item_id=f'i{i}', clean_mass=0.9, protected_mass=0.9 * COLLAPSE_RATIO / 2) for i in range(100)
    ]
    d = diagnose_format(rows, freegen_agreement={'clean': 0.99, 'protected': 0.99})
    assert d['status'] == 'flagged'
    assert any('collapsed' in f for f in d['flags'])


def test_format_diagnostic_passes_a_healthy_run() -> None:
    d = diagnose_format(
        [item(item_id=f'i{i}') for i in range(100)], freegen_agreement={'clean': 0.99, 'protected': 0.98}
    )
    assert d['status'] == 'ok', d['flags']


def test_format_diagnostic_flags_unmeasured_free_generation() -> None:
    d = diagnose_format([item(item_id=f'i{i}') for i in range(100)])
    assert d['status'] == 'flagged' and any('free-generation' in f for f in d['flags'])


def test_format_diagnostic_stops_on_low_free_generation_agreement() -> None:
    try:
        diagnose_format(
            [item(item_id=f'i{i}') for i in range(100)], freegen_agreement={'clean': 0.99, 'protected': 0.80}
        )
    except ValueError:
        return
    raise AssertionError('agreement below the threshold must stop the run')


def test_format_diagnostic_rejects_non_finite_mass() -> None:
    try:
        diagnose_format(
            [item(item_id=f'i{i}', clean_mass=float('nan')) for i in range(10)],
            freegen_agreement={'clean': 0.99, 'protected': 0.99},
        )
    except ValueError:
        return
    raise AssertionError('non-finite mass must abort, not report ok')


def test_validate_rejects_a_self_contradicting_row() -> None:
    bad = item()
    bad.protected_correct = False  # contradicts protected_pred == gold
    try:
        validate([bad])
    except ValueError:
        return
    raise AssertionError('a row contradicting its own predictions must be rejected')


def test_clopper_pearson_does_not_overflow_at_realistic_sizes() -> None:
    """comb(10000, 200) cannot convert to float; the log-space sum must handle it."""
    v = clopper_pearson_upper(200, 10000)
    assert 0.02 < v < 0.03, v
    # The cutoff is where the bound crosses 1%: the largest x that still certifies.
    assert clopper_pearson_upper(119, 13866) < 0.01  # 13,866 clean-correct -> 119 passes
    assert clopper_pearson_upper(120, 13866) > 0.01  # ...and 120 does not
    assert clopper_pearson_upper(4, 1000) < 0.01  # 1,000 clean-correct -> 4 passes
    assert clopper_pearson_upper(5, 1000) > 0.01  # ...and 5 does not
    # Same observed RATE, wildly different verdicts -- which is why quoting a regression
    # allowance without its denominator is meaningless.


def test_clopper_pearson_rejects_invalid_counts() -> None:
    for x, n in ((5, 3), (-1, 10), (1, -2)):
        try:
            clopper_pearson_upper(x, n)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f'must reject x={x}, n={n}')


def test_label_change_rate_sees_wrong_to_different_wrong() -> None:
    """A changed-but-still-wrong answer is invisible to the regression rate by design."""
    rows = [item(item_id=f'i{i}', gold=0, clean_pred=1, protected_pred=2) for i in range(10)]
    s = summarise(rows)
    assert s['regressions'] == 0 and s['label_change_rate'] == 1.0


def test_validate_rejects_a_prediction_that_contradicts_its_scores() -> None:
    """Scores select option 1 while the row claims option 0: the row must be rejected."""
    bad = item(gold=0)
    bad.protected_scores = [0.0, 9.0, 0.0, 0.0]
    try:
        validate([bad])
    except ValueError as e:
        assert 'contradicts its scores' in str(e)
        return
    raise AssertionError('a prediction contradicting its own scores must be rejected')


def test_validate_rejects_duplicate_item_ids() -> None:
    """Guards against 299 copies of one row producing a legitimate-looking 0.00997 bound."""
    try:
        validate([item(item_id='dup'), item(item_id='dup')])
    except ValueError:
        return
    raise AssertionError('duplicate item ids must be rejected')


def test_validate_rejects_impossible_masses_and_counts() -> None:
    for kw in (
        {'clean_mass': 2.0},
        {'protected_mass': -0.1},
        {'tokens_changed': 11, 'content_tokens': 10},
        {'prompt_tokens': -1},
    ):
        try:
            validate([item(**kw)])
        except ValueError:
            continue
        raise AssertionError(f'must reject {kw}')


def test_exact_logprob_vector_with_distinct_values() -> None:
    """Reversing the log-probability vector must be caught.

    With four EQUAL logits a reversal would be invisible, so the values here are distinct.
    """
    v = logits_with([3.0, 1.0, -2.0, 0.5])
    scores, logprobs, _ = score_options(FakeModel(v), None, LABEL_IDS)
    assert scores == [3.0, 1.0, -2.0, 0.5], f'ranking scores are wrong: {scores}'
    expect = torch.log_softmax(torch.tensor(v, dtype=torch.float64), 0)
    want = [expect[i[0]].item() for i in LABEL_IDS]
    assert all(abs(a - b) < 1e-12 for a, b in zip(logprobs, want)), 'log-probs are wrong'
    assert logprobs != list(reversed(logprobs)), 'fixture must not be symmetric'
    # values are [3.0, 1.0, -2.0, 0.5] -> index order 0 > 1 > 3 > 2
    assert logprobs[0] > logprobs[1] > logprobs[3] > logprobs[2], 'ordering must match'


def test_exact_candidate_mass_value() -> None:
    """Reporting half the real mass must be caught."""
    v = logits_with([1.0, 1.0, 1.0, 1.0])
    v[7] = 1.0
    _, _, mass = score_options(FakeModel(v), None, LABEL_IDS)
    lp = torch.log_softmax(torch.tensor(v, dtype=torch.float64), 0)
    want = sum(lp[i[0]].exp().item() for i in LABEL_IDS)
    assert abs(mass - want) < 1e-12, f'mass {mass} != exact {want}'
    assert abs(mass - 4 / 5) < 1e-9, 'five equal logits -> the four candidates hold 4/5'


def test_alternate_label_mapping_is_actually_used() -> None:
    """Hardcoded label ids [41,17,93,64] must be caught by a different mapping."""
    other = [[11], [72], [5], [108]]
    v = logits_with([7.0, 6.0, 5.0, 4.0], ids=other)
    scores, _, _ = score_options(FakeModel(v), None, other)
    assert scores == [7.0, 6.0, 5.0, 4.0], f'ignored the supplied ids: {scores}'


def test_score_options_rejects_malformed_continuations() -> None:
    v = logits_with([1.0, 2.0, 3.0, 4.0])
    for bad in ([[41], [17], [93]], [[41], [17], [93], [-1]], [[41]] * 4):
        try:
            score_options(FakeModel(v), None, bad)
        except ValueError:
            continue
        raise AssertionError(f'must reject continuations {bad}')


def test_protocol_hash_covers_the_confidence_bound() -> None:
    """Replacing clopper_pearson_upper with a stub must change the hash."""
    import evaluation.mmlu_scorer as m

    before = m.protocol_hash()
    original = m.clopper_pearson_upper
    m.clopper_pearson_upper = lambda x, n, alpha=0.05: 0.0
    try:
        assert m.protocol_hash() != before, 'the hash must cover the confidence bound'
    finally:
        m.clopper_pearson_upper = original
    assert m.protocol_hash() == before


def test_freegen_agreement_must_be_per_arm_and_finite() -> None:
    rows = [item(item_id=f'i{i}') for i in range(10)]
    for bad in (
        0.99,
        {'clean': 0.99},
        {'clean': float('nan'), 'protected': 0.99},
        {'clean': 1.5, 'protected': 0.99},
    ):
        try:
            diagnose_format(rows, freegen_agreement=bad)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f'must reject freegen_agreement={bad!r}')


def test_clopper_pearson_rejects_bad_alpha() -> None:
    for al in (1.5, -0.05, 0.0, 1.0, float('nan')):
        try:
            clopper_pearson_upper(0, 10, alpha=al)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f'must reject alpha={al!r}')


def test_clopper_pearson_correct_above_the_observed_rate() -> None:
    """A root bracket that starts at x/n would exclude solutions below it."""
    v = clopper_pearson_upper(1, 10, alpha=0.9)
    assert abs(v - 0.05452862) < 1e-6, v


def test_predict_ties_that_exclude_option_a() -> None:
    """A tie-break that always returns 0 must be caught.

    A tie that INCLUDES A, such as [2,2,1,0], cannot tell that apart from a correct
    lowest-index tie-break. A scorer that always returns 0 accepts a real regression as
    no change, so these ties exclude A.
    """
    assert predict([0.0, 5.0, 5.0, 1.0])[:2] == (1, True)  # tie between B and C
    assert predict([0.0, 1.0, 5.0, 5.0])[:2] == (2, True)  # tie between C and D
    assert predict([1.0, 1.0, 1.0, 9.0])[:2] == (3, False)  # D alone wins
    assert predict([0.0, 0.0, 0.0, 0.0])[:2] == (0, True)  # four-way tie -> A


def test_validate_catches_a_regression_hidden_by_a_bad_tie_break() -> None:
    """Protected scores [0,5,5,1], correct answer B: B wins the tie, a stored A is wrong."""
    r = item(gold=1, clean_pred=1)
    r.protected_scores = [0.0, 5.0, 5.0, 1.0]
    r.protected_pred, r.tie_protected, r.protected_margin = 1, True, 0.0
    r.protected_correct = True
    r.transition = transition(True, True)
    r.label_changed = False
    validate([r])  # consistent: B wins the tie
    r.protected_pred = 0  # what a broken tie-break would store
    try:
        validate([r])
    except ValueError:
        return
    raise AssertionError('a wrong tie-break winner must be rejected')


def test_label_token_ids_maps_labels_in_order() -> None:
    """Reversing the label order must be caught."""

    class Tok:
        def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
            return {'input_ids': [{' A': 41, ' B': 17, ' C': 93, ' D': 64}[text]]}

    ids, length = label_token_ids(Tok())
    assert length == 1
    assert ids == [[41], [17], [93], [64]], f'label order is wrong: {ids}'


def test_label_token_ids_stops_on_unequal_lengths() -> None:
    class Tok:
        def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
            return {'input_ids': [1, 2] if text == ' C' else [1]}

    try:
        label_token_ids(Tok())
    except ValueError:
        return
    raise AssertionError('unequal continuation lengths must stop the run')


def test_validate_accepts_mass_at_floating_point_one() -> None:
    """fp64 summation legitimately gives 1.0000000000000002; that must pass, 1.5 must not."""
    validate([item(item_id='a', clean_mass=1.0000000000000002, protected_mass=1.0000000000000002)])
    try:
        validate([item(item_id='b', clean_mass=1.5)])
    except ValueError:
        return
    raise AssertionError('a mass of 1.5 must still be rejected')


def test_validate_rejects_bools_as_counts() -> None:
    try:
        validate([item(tokens_changed=True)])
    except ValueError:
        return
    raise AssertionError('bool must not pass as an integer count')


def test_diagnose_format_validates_its_rows() -> None:
    """A row with mass 2.0 must be rejected, never reach status 'ok'."""
    try:
        diagnose_format(
            [item(item_id='x', clean_mass=2.0)], freegen_agreement={'clean': 0.99, 'protected': 0.99}
        )
    except ValueError:
        return
    raise AssertionError('diagnose_format must validate rows')


def test_freegen_rejects_booleans() -> None:
    rows = [item(item_id=f'i{i}') for i in range(5)]
    try:
        diagnose_format(rows, freegen_agreement={'clean': True, 'protected': True})
    except (ValueError, TypeError):
        return
    raise AssertionError('True must not pass as an agreement fraction')


def test_protocol_hash_sees_a_mutated_default() -> None:
    """Changing alpha via __defaults__ moves the bound 0.00997 -> 0.00232; the hash must change."""
    import evaluation.mmlu_scorer as m

    before = m.protocol_hash()
    saved = m.clopper_pearson_upper.__defaults__
    m.clopper_pearson_upper.__defaults__ = (0.5,)
    try:
        assert m.protocol_hash() != before, 'a mutated default must change the hash'
    finally:
        m.clopper_pearson_upper.__defaults__ = saved
    assert m.protocol_hash() == before


def test_protocol_hash_sees_a_dataclass_hook() -> None:
    """An ItemResult.__post_init__ that overwrites protected_mass must change the hash."""
    import evaluation.mmlu_scorer as m

    before = m.protocol_hash()
    m.ItemResult.__post_init__ = lambda self: None
    try:
        assert m.protocol_hash() != before, 'a dataclass hook must change the hash'
    finally:
        del m.ItemResult.__post_init__
    assert m.protocol_hash() == before


def test_all_negative_logits_keep_their_order() -> None:
    """A clamp to zero on the ranking scores must be caught.

    Real logits are routinely negative. With candidates [-4,-1,-3,-2] the answer is B; a
    scorer clamping at zero returns four zeros, predicts A and reports a tie -- hiding a
    genuine regression whenever gold is A.
    """
    v = logits_with([-4.0, -1.0, -3.0, -2.0])
    scores, _, _ = score_options(FakeModel(v), None, LABEL_IDS)
    assert scores == [-4.0, -1.0, -3.0, -2.0], f'negative logits altered: {scores}'
    idx, tie, _ = predict(scores)
    assert (idx, tie) == (1, False), 'the largest negative logit must win, with no tie'


def test_closely_spaced_logits_are_not_rounded_together() -> None:
    """Casting through bf16 before ranking must be caught.

    bf16 has ~8 bits of mantissa, so 1.0 and 1.001 collapse to the same value and A ties B.
    """
    v = logits_with([1.0, 1.001, 0.0, -1.0])
    scores, _, _ = score_options(FakeModel(v), None, LABEL_IDS)
    idx, tie, margin = predict(scores)
    assert (idx, tie) == (1, False), f'lost precision: {scores}'
    assert 0 < margin < 0.01
    assert torch.tensor(1.0).to(torch.bfloat16) == torch.tensor(1.001).to(torch.bfloat16), (
        'fixture is only meaningful if bf16 really does collapse them'
    )


def test_mass_is_normalised_over_the_full_vocabulary_not_the_candidates() -> None:
    """Normalising over the four candidates alone would always give mass 1.0."""
    v = logits_with([1.0, 1.0, 1.0, 1.0])
    for i in (7, 8, 9):
        v[i] = 3.0  # substantial non-candidate mass
    _, logprobs, mass = score_options(FakeModel(v), None, LABEL_IDS)
    assert mass < 0.5, f'mass {mass} suggests candidate-only normalisation'
    lp = torch.log_softmax(torch.tensor(v, dtype=torch.float64), 0)
    assert abs(mass - sum(lp[i[0]].exp().item() for i in LABEL_IDS)) < 1e-12


def test_label_token_ids_with_a_second_mapping() -> None:
    """One successful mapping is not enough to pin the function's behaviour."""

    class Tok:
        def __init__(self, m: dict[str, int]) -> None:
            self.m = m

        def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
            return {'input_ids': [self.m[text]]}

    ids, _ = label_token_ids(Tok({' A': 11, ' B': 72, ' C': 5, ' D': 108}))
    assert ids == [[11], [72], [5], [108]], ids
    ids, _ = label_token_ids(Tok({' A': 900, ' B': 901, ' C': 902, ' D': 903}))
    assert ids == [[900], [901], [902], [903]], ids


def test_mass_flag_share_boundary_clean_arm_only() -> None:
    """Exactly 5% on the CLEAN arm must flag: this pins `>=` versus `>`.

    Only the clean arm is low here. If BOTH masses were low, the protected branch would
    flag the run anyway and a wrong clean comparison would go unnoticed.
    """
    rows = [item(item_id=f'lo{i}', clean_mass=0.001, protected_mass=0.9) for i in range(5)]
    rows += [item(item_id=f'ok{i}') for i in range(95)]
    d = diagnose_format(rows, freegen_agreement={'clean': 0.99, 'protected': 0.99})
    assert d['low_mass_clean'] == 5 and d['low_mass_protected'] == 0
    assert d['status'] == 'flagged', 'exactly 5% low-mass clean items must flag the run'


def test_mass_flag_share_boundary_protected_arm_only() -> None:
    """Mirror of the clean case, isolating the protected branch.

    Masses are chosen so ONLY that branch can fire: clean 0.05 is above MASS_FLOOR so the
    clean branch stays quiet, protected 0.009 is below it, and 0.009 > 0.1 x 0.05 so the
    collapse branch stays quiet too. Masses of 0.9 / 0.001 would trip collapse and hide a
    wrong comparison here.
    """
    rows = [item(item_id=f'lo{i}', clean_mass=0.05, protected_mass=0.009) for i in range(5)]
    rows += [item(item_id=f'ok{i}') for i in range(95)]
    d = diagnose_format(rows, freegen_agreement={'clean': 0.99, 'protected': 0.99})
    assert d['low_mass_protected'] == 5 and d['low_mass_clean'] == 0
    assert d['mass_collapsed'] == 0, 'the collapse branch must not fire in this fixture'
    assert d['status'] == 'flagged', 'exactly 5% low-mass protected items must flag the run'


def test_collapse_share_boundary_is_inclusive() -> None:
    """The collapse branch needs its own boundary case, isolated from the mass floors."""
    rows = [item(item_id=f'c{i}', clean_mass=0.9, protected_mass=0.05) for i in range(5)]
    rows += [item(item_id=f'ok{i}') for i in range(95)]
    d = diagnose_format(rows, freegen_agreement={'clean': 0.99, 'protected': 0.99})
    assert d['low_mass_clean'] == 0 and d['low_mass_protected'] == 0
    assert d['mass_collapsed'] == 5 and d['status'] == 'flagged'


def test_mass_flag_share_just_below_the_boundary_does_not_flag() -> None:
    rows = [item(item_id=f'lo{i}', clean_mass=0.001, protected_mass=0.9) for i in range(4)]
    rows += [item(item_id=f'ok{i}') for i in range(96)]
    d = diagnose_format(rows, freegen_agreement={'clean': 0.99, 'protected': 0.99})
    assert d['status'] == 'ok', f'4% must not flag: {d["flags"]}'


def test_validate_rejects_content_tokens_exceeding_prompt_tokens() -> None:
    """content_tokens may not exceed prompt_tokens; equal counts are allowed."""
    try:
        validate([item(content_tokens=21, prompt_tokens=20)])
    except ValueError:
        pass
    else:
        raise AssertionError('content_tokens may not exceed prompt_tokens')
    validate([item(content_tokens=20, prompt_tokens=20)])


def test_validate_rejects_bool_masses_and_float_flags() -> None:
    for kw in (
        {'clean_mass': True, 'protected_mass': True},
        {'clean_correct': 1.0},
        {'tie_clean': 1},
        {'clean_margin': -0.5},
    ):
        try:
            validate([item(**kw)])
        except (ValueError, TypeError):
            continue
        raise AssertionError(f'must reject {kw}')


def test_validate_rejects_missing_identifiers() -> None:
    for kw in ({'item_id': None}, {'item_id': ''}, {'subject': None}):
        try:
            validate([item(**kw)])
        except (ValueError, TypeError):
            continue
        raise AssertionError(f'must reject {kw}')


def test_validate_rejects_negative_mass() -> None:
    """A symmetric tolerance would accept -5e-10, which no correct sum produces."""
    try:
        validate([item(clean_mass=-5e-10)])
    except ValueError:
        return
    raise AssertionError('a negative mass must be rejected')


def test_protocol_hash_sees_hook_behaviour_not_just_presence() -> None:
    """Two different __post_init__ bodies must produce different hashes."""
    import evaluation.mmlu_scorer as m

    def hook_a(self) -> None:
        self.protected_mass = 0.0

    def hook_b(self) -> None:
        self.protected_mass = 0.5

    before = m.protocol_hash()
    m.ItemResult.__post_init__ = hook_a
    try:
        ha = m.protocol_hash()
        m.ItemResult.__post_init__ = hook_b
        hb = m.protocol_hash()
    finally:
        del m.ItemResult.__post_init__
    assert ha != before, 'adding a hook must change the hash'
    assert ha != hb, 'two different hook bodies must hash differently'
    assert m.protocol_hash() == before


def test_protocol_hash_is_stable_across_processes() -> None:
    """Three fresh processes must produce the same hash.

    ItemResult's dataclass-generated __eq__ has no readable source. A repr() fallback
    would embed a memory address: no run would be reproducible, and resume would reject
    its own rows.
    """
    code = (
        'import sys; sys.path.insert(0, "."); '
        'from evaluation.mmlu_scorer import protocol_hash; print(protocol_hash())'
    )
    seen = {
        subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        ).stdout.strip()
        for _ in range(3)
    }
    assert len(seen) == 1, f'hash differs across processes: {seen}'
    assert '' not in seen, 'hash computation failed in the subprocess'


def test_scoring_preserves_the_sign_of_the_embeddings() -> None:
    """`embeds.abs()` on the input path must be caught.

    With only NON-NEGATIVE embeddings, sign destruction is invisible. Real embeddings are
    signed. An echo model given [-4,-1,-3,-2] must select
    B; under abs() it sees [4,1,3,2] and selects A -- a wrong answer, silently.

    This is the INPUT-path counterpart of the clamp_min(0) case on the RANKING path
    (test_all_negative_logits_keep_their_order).
    """
    e = torch.tensor([[[-4.0, -1.0, -3.0, -2.0]]])
    scores, _, _ = score_options(EchoModel(), e, LABEL_IDS)
    assert scores == [-4.0, -1.0, -3.0, -2.0], f'signs were altered: {scores}'
    assert predict(scores)[0] == 1, 'the least-negative candidate must win'
    # and a mixed-sign case, so neither abs() nor a clamp can reproduce the ordering
    mixed = torch.tensor([[[-2.0, 3.0, -5.0, 1.0]]])
    mscores, _, _ = score_options(EchoModel(), mixed, LABEL_IDS)
    assert mscores == [-2.0, 3.0, -5.0, 1.0], f'signs were altered: {mscores}'
    assert predict(mscores)[0] == 1
    assert [abs(x) for x in mscores] != mscores, 'fixture must contain negatives'


def test_scoring_depends_on_the_supplied_embeddings() -> None:
    """Zeroing inputs_embeds must be caught.

    FakeModel ignores its input and the other scoring fixtures pass embeds=None, so they
    cannot show that the scores depend on the tensor supplied. A scorer that discards the
    prompt makes clean and protected identical and erases every regression.
    """
    e = torch.tensor([[[1.0, 5.0, 2.0, 3.0]]])
    scores, _, _ = score_options(EchoModel(), e, LABEL_IDS)
    assert scores == [1.0, 5.0, 2.0, 3.0], f'embeddings were not used: {scores}'
    assert predict(scores)[0] == 1
    zeros = torch.zeros_like(e)
    zscores, _, _ = score_options(EchoModel(), zeros, LABEL_IDS)
    assert zscores != scores, 'zeroed embeddings must not give the same scores'


def test_mass_floor_boundary_is_strict() -> None:
    """`<=` instead of `<` on MASS_FLOOR must be caught.

    The protocol says M_i < 0.01. An item sitting exactly on the floor is NOT low-mass.
    """
    rows = [item(item_id=f'e{i}', clean_mass=MASS_FLOOR, protected_mass=0.9) for i in range(10)]
    rows += [item(item_id=f'ok{i}') for i in range(90)]
    d = diagnose_format(rows, freegen_agreement={'clean': 0.99, 'protected': 0.99})
    assert d['low_mass_clean'] == 0, 'mass exactly at the floor must not count as low'
    assert d['status'] == 'ok'
    below = [item(item_id=f'b{i}', clean_mass=MASS_FLOOR - 1e-12, protected_mass=0.9) for i in range(10)] + [
        item(item_id=f'ok{i}') for i in range(90)
    ]
    d2 = diagnose_format(below, freegen_agreement={'clean': 0.99, 'protected': 0.99})
    assert d2['low_mass_clean'] == 10, 'just below the floor must count'


class ContextModel(torch.nn.Module):
    """A causal toy whose every earlier input affects later logits."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(6, 6)
        self.embedding.weight.data.copy_(torch.eye(6))
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> SimpleNamespace:
        e = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        e = self.layers[0](e)
        return SimpleNamespace(logits=e.cumsum(1))


def test_all_prompt_positions_reach_model_and_affect_prediction() -> None:
    model = ContextModel()
    e = torch.tensor([[[0.0, 8.0, 0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    scores, _, _ = score_options(model, e, [[0], [1], [2], [3]])
    assert scores == [3.0, 8.0, 0.0, 0.0]
    assert predict(scores)[0] == 1
    truncated, _, _ = score_options(model, e[:, -1:], [[0], [1], [2], [3]])
    assert predict(truncated)[0] == 0


def test_multitoken_sums_correct_causal_positions_and_clean_suffixes() -> None:
    from evaluation.mmlu_contract import oracle_scores

    model = ContextModel()
    ids = [4, 5]
    e = model.embedding(torch.tensor([ids]))
    # First tokens identical: scoring only the first token falsely returns four ties.
    labels = [[0, 0, 1], [0, 1, 0], [0, 2, 3], [0, 3, 2]]
    scores, lp, mass = score_options(model, e, labels)
    expected = []
    for candidate in labels:
        counts = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        total = 0.0
        for token in candidate:
            total += counts[token] - math.log(sum(math.exp(v) for v in counts))
            counts[token] += 1
        expected.append(total)
    assert all(abs(a - b) < 1e-12 for a, b in zip(scores, expected))
    assert scores == lp
    assert abs(mass - sum(math.exp(v) for v in expected)) < 1e-12
    assert scores[0] != scores[1]
    assert all(abs(a - b) < 1e-12 for a, b in zip(scores, oracle_scores(model, ids, labels)))
    protected = e + torch.tensor([[[0.0, 2.0, 0.0, 0.0, 0.0, 0.0]]])
    ps, _, _ = score_options(model, protected, labels)
    assert ps != scores
    assert all(abs(a - b) < 1e-12 for a, b in zip(ps, oracle_scores(model, ids, labels, protected)))


def test_subject_bound_uses_clean_correct_weights_and_empty_subjects() -> None:
    rows = [item(item_id='a', subject='one', protected_pred=1)]
    rows += [item(item_id=f'b{i}', subject='two') for i in range(3)]
    rows += [item(item_id='c', subject='zero', clean_pred=1, protected_pred=1)]
    summary = summarise(rows)
    allowance = math.sqrt(math.log(20) / 2 * ((1 / 4) ** 2 + (3 / 4) ** 2))
    assert abs(summary['subject_block_allowance'] - allowance) < 1e-12
    assert summary['per_subject']['zero']['regression_rate'] is None
    assert summary['per_subject']['one']['regressions'] == 1
    assert summary['regression_lower_95'] < 0.25 < summary['regression_upper_95']
    assert summary['subject_block_upper_95'] == min(1.0, 0.25 + allowance)
    empty = summarise([])
    assert empty['subject_block_upper_95'] is None
    assert empty['regression_lower_95'] is None


def test_format_exact_collapse_ratio_does_not_count_as_collapsed() -> None:
    rows = [item(clean_mass=0.5, protected_mass=0.5 * COLLAPSE_RATIO)]
    assert diagnose_format(rows, {'clean': 1.0, 'protected': 1.0})['mass_collapsed'] == 0


def test_obfuscation_both_precisions_and_estimands() -> None:
    from evaluation.mmlu_scorer import obfuscation_summary

    rows = [
        item(item_id='a', tokens_changed=1, tokens_changed_bf16=2, content_tokens=2),
        item(item_id='b', tokens_changed=1, tokens_changed_bf16=0, content_tokens=8),
    ]
    got = obfuscation_summary(rows)
    assert got['macro_pct'] == 31.25 and got['pooled_pct'] == 20
    assert got['macro_pct_bf16'] == 50 and got['pooled_pct_bf16'] == 20


def test_protected_mass_floor_boundary_is_strict_too() -> None:
    at = diagnose_format([item(clean_mass=0.05, protected_mass=MASS_FLOOR)], {'clean': 1.0, 'protected': 1.0})
    below = diagnose_format(
        [item(clean_mass=0.05, protected_mass=MASS_FLOOR - 1e-12)], {'clean': 1.0, 'protected': 1.0}
    )
    assert at['low_mass_protected'] == 0 and at['mass_collapsed'] == 0
    assert below['low_mass_protected'] == 1


def test_hash_covers_handwritten_override_of_generated_method() -> None:
    import evaluation.mmlu_scorer as m

    before = m.protocol_hash()
    original = m.ItemResult.__init__

    def changed(self) -> None:
        self.protected_mass = 0.5

    try:
        m.ItemResult.__init__ = changed
        assert m.protocol_hash() != before
    finally:
        m.ItemResult.__init__ = original
    assert m.protocol_hash() == before


def test_unreadable_handwritten_hook_fails_closed() -> None:
    import pytest

    import evaluation.mmlu_scorer as m

    namespace = {}
    exec('def hook(self): pass', namespace)
    original = m.ItemResult.__init__
    try:
        m.ItemResult.__init__ = namespace['hook']
        with pytest.raises(RuntimeError, match='cannot read'):
            m.protocol_hash()
    finally:
        m.ItemResult.__init__ = original
