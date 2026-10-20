# MMLU scoring protocol

This file defines how `scripts/evaluate.py` measures what the input-embedding transform does to the
multiple-choice answers of the frozen model. The code is the executable definition:
`evaluation/mmlu_scorer.py` (user message, option scoring, row checks, statistics, format diagnostic),
`evaluation/mmlu_contract.py` (run identity, subset selection, answer parser, scoring oracle,
decoder-input probe) and `scripts/evaluate.py` (prompt string, transform, controls, resume, outputs).
If this text and the code disagree, the code defines what a run measured. Every run hashes this file
into its manifest (section 9), so a change here starts a new run.

Contents: 1 What is measured, 2 Data and splits, 3 Prompt, 4 Scoring, 5 What the transform touches,
6 Format validation, 7 Controls, 8 Statistics, 9 Outputs, 10 Limits.

Model: `Qwen/Qwen3.8-27B`, model revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, read from
`QWEN_MODEL_DIR` (default `models/Qwen3.8-27B`).

## 1. What is measured

Every question is scored twice on the same item: the **clean arm** uses the prompt's own token
embeddings `E`, the **protected arm** uses the delivered embeddings `Z` (sections 4 and 5). Each item
gets one transition (row field `transition`), clean outcome first: `right_right`, `right_wrong`,
`wrong_right` or `wrong_wrong`.

**Break rate (primary).** Of the questions the clean model answers correctly, the share that the
protected model answers wrong: `right_wrong / (right_right + right_wrong)`. In `report.json` this is
`summary.regression_rate`, with count `regressions` and denominator `clean_correct`.

**Repairs** (`wrong_right`, field `improvements`) are counted and reported separately and never
subtract from breaks. A net accuracy change would hide breaks: 20 breaks and 19 repairs look like
almost no change.

Context only, never a gate: clean and protected accuracy; retention `1 - x/n + y/n` (`x` breaks, `y`
repairs, `n` clean-correct items); and the label-change rate, the share of all items whose predicted
label changes. It also counts a wrong answer that becomes a different wrong answer, which the break
rate does not see.

**Hiding rate (diagnostic only).** The share of transformed prompt tokens whose nearest vocabulary
embedding is no longer the original token (section 5). It is reported beside every break rate, enters
no gate, count or bound, and is not a privacy measure (section 10).

## 2. Data and splits

| file in `data/` | items | use |
|---|---:|---|
| `mmlu_dev.jsonl` | 1,490 | the scored set, 57 subjects |
| `mmlu_format_validation.jsonl` | 285 | items used to choose the prompt; kept out of validation |
| `mmlu_format_validation_r5.jsonl` | 276 | the format-validation subset (section 6) |
| `manifest.json` | - | SHA-256 of each file, the selection rules, the upstream file hashes |

Each row holds `i` (the item's index in the upstream validation file), `question`, `choices` (four
strings), `answer` (0-3), `subject`, and three 32-hex-digit keys: `key` (question with the options in
upstream order), `gkey` (question with the options sorted) and `stem` (question alone). The item id in
every output is `mmlu_dev:<subject>:<i>`. Subjects hold between 5 and 170 items.

**Source.** `cais/mmlu`, config `all`, upstream `validation` split: 1,531 questions. The upstream
`test` split is never used for scoring here. The upstream `dev` split (few-shot examples) is not used:
the prompt is zero-shot.

**Carving rule.** Text is normalised by Unicode NFKC, casefold, curly quotes to an ASCII apostrophe,
whitespace runs to one space, and strip. A key is the first 32 hex digits of the SHA-256 of the
normalised text. Items are grouped on two keys at once: `gkey` catches one item with reordered
options, and `stem` catches one question with different distractors. The carve does two passes:

1. Within each upstream split, keep the first item of each `gkey` and drop later copies.
2. Walk the upstream test split, then the validation split, in file order. Keep an item only if no
   item kept before it holds its `gkey` or its `stem`. A kept item holds both of its keys.

This removes 41 validation items: 2 in the first pass (same question and option set as an earlier
validation item) and 39 in the second (38 share a key with a test item; 1 shares its stem with an
earlier validation item). No item of `mmlu_dev.jsonl` shares either key with an upstream test item or
with another `mmlu_dev` item. Paraphrases and translations are not detected.

`scripts/evaluate.py` stops unless the SHA-256 of `data/mmlu_dev.jsonl` equals
`splits.mmlu_dev.sha256` in `data/manifest.json`, and stops if either format file differs from the
subset it rebuilds (section 6).

## 3. Prompt

Zero-shot. The user message is the question, the four options and one instruction line, joined by
single newlines, with no trailing newline:

```text
{question}
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}
Respond with only the letter of the correct option: A, B, C, or D.
```

The message goes through `tok.apply_chat_template([{'role': 'user', 'content': message}],
tokenize=False, add_generation_prompt=True, enable_thinking=False)`. The literal text `Answer:` (no
space, no newline) is then appended as an assistant prefill, and the whole string is tokenized once
with `add_special_tokens=False`. With the pinned tokenizer the full prompt string is:

```text
<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:
```

Each run logs this string for its first item as `rendered_prefix` in its manifest. Likelihood scoring
and format-validation generation use the same string and add nothing to it.

## 4. Scoring

**Candidates.** `" A"`, `" B"`, `" C"` and `" D"`, tokenized with `add_special_tokens=False`. The run
stops if they differ in token count, are empty or are not distinct. Equal length is why no length
normalisation is applied. With the pinned tokenizer each label is one token.

**Single-token labels (the pinned model).** One forward pass over the prompt embeddings with
`use_cache=False`. The logits at the last prompt position (right after `Answer:`) are cast to fp64,
and the ranking score of each option is its **raw logit**. The four candidates share one prefix, so
the softmax normaliser is common to all of them and is not computed for ranking; this avoids ties made
by rounding four separate log-probabilities. An fp64 log-softmax over the full vocabulary gives the
diagnostics only: per-option log-probabilities and the candidate mass `M` (summed probability of the
four labels).

**Multi-token labels (other tokenizers, and control C1a).** Each candidate gets its own uncached
teacher-forced pass: the clean embeddings of all but its last token are appended to the (possibly
transformed) prompt, and the fp64 log-probabilities of its tokens are summed. The sum is both score
and log-probability. No cache is shared between candidates.

**Prediction and ties.** The prediction is the highest score. An exact tie for the top goes to the
lowest option index and sets `tie_clean` or `tie_protected`. The margin is the best score minus the
second best. A non-finite score stops the run; no item is dropped.

**Precision.** Model weights bf16 (`sdpa` attention, eval mode). The transform runs in fp32 on `E`
cast to fp32; the blend is computed in fp32 and cast to bf16 before the model sees it. Scoring
arithmetic is fp64. `torch.backends.cuda.matmul.allow_tf32 = False`, batch size 1, seed 20260911, and
the transform runs with `stochastic=False` (its mean output, no sampled noise).

**Alpha.** The delivered input is `Z = (1 - alpha) * E + alpha * T(E)` with `0 <= alpha <= 1`
(default 1). This form makes both ends exact: alpha = 0 gives `E` bitwise and alpha = 1 gives `T(E)`
bitwise (control C2). Each alpha is a different system with a different run hash.

## 5. What the transform touches

**The prompt only.** The transform runs once per item on the full prompt, before any candidate is
scored, and the same `Z` serves all four options. Candidate tokens are never transformed and never
reach the transform's input; otherwise the protected prompt would depend on the option being scored.
In format-validation generation, generated tokens are fed back as clean embeddings.

**Content and pass-through positions.** The content mask is `token_id not in tok.all_special_ids`. At
content positions the transform output replaces `E`. At the other (pass-through) positions the
transform returns `E` unchanged, but its context encoder still reads them. The mask leaves out only
registered special-token ids, not the chat scaffolding as text. With the pinned tokenizer exactly one
position per MMLU prompt passes through: the `<|im_end|>` that closes the user message.
`<|im_start|>`, the role words, the thinking tags, the newlines, the instruction and the `Answer:`
prefill are content and are transformed.

**Hiding rate.** For each content position the decoder takes the vocabulary row with the highest
cosine similarity to the delivered vector (both L2-normalised in fp32; ties go to the lowest
vocabulary id). The token is *changed* if that row is not the original token. Rows record
`tokens_changed` (fp32 blend), `tokens_changed_bf16` (blend cast to bf16, which the model receives) and
`content_tokens`. Two estimates are reported, overall and per subject, and never mixed: **macro**
(`macro_pct`, `macro_pct_bf16`), the mean over items of the per-item percentage, leaving out items
with no content tokens (`zero_content_items`); and **pooled** (`pooled_pct`, `pooled_pct_bf16`), total
changed tokens over total content tokens.

**Length.** No length filter and no truncation. An item needs its prompt length plus
`max(label tokens, 32)` positions in the format subset, or plus `max(label tokens, 2)` otherwise (2
covers the two-token C1a fixture). If any item needs more than `max_position_embeddings`, the run
writes `exclusions.json` (item id, prompt tokens, required tokens) and stops before scoring.

## 6. Format validation

Option likelihoods give an answer even when the model would never emit a bare label, and a transform
can collapse the probability of all four labels while keeping their order. This section checks both.

**Subsets.** Both come from `mmlu_dev.jsonl`, which must have all 57 subjects, with one ordering:
within each subject (subjects in lexical order), items are sorted by ascending SHA-256 of the JSON
text `[20260911, "mmlu_dev:<subject>:<i>"]` (Python `json.dumps`, default separators).

- `mmlu_format_validation.jsonl`: the first five items of each subject, 285 items. The prompt of
  section 3 was chosen on these items with clean-only tests, so they are not used for validation.
  They stay in the scored set.
- `mmlu_format_validation_r5.jsonl`: the same order without those 285 items, then the first five
  remaining items per subject. A subject with fewer gives what remains: `college_chemistry` and
  `high_school_computer_science` give 3 each, `college_physics` none. Result: 276 items, 56 subjects.

**Generation.** Each validation item is scored in both arms with the section 7 controls. Then the model
generates greedily in each arm from that arm's delivered prompt: KV cache on, at most 32 new tokens,
`argmax` at each position, stop at an EOS id (generation-config EOS ids plus the tokenizer EOS id).
The text is decoded with `skip_special_tokens=True`; a generation without EOS within 32 tokens is not
parsed. The parser accepts only a full match of `\s*([ABCD])\.?\s*`: one uppercase label, at most one
period, whitespace at either end. Explanations, several labels, lowercase, empty text and the
32-token cap are misses. No item leaves the denominator.

**Gate (stops the run).** Agreement in an arm is the share of the 276 items whose parsed label equals
that arm's likelihood prediction. Each arm needs agreement >= 0.95, that is at least 263 of 276,
checked per arm and never pooled. If either arm fails, the run stops before it scores any other item.
A tie for the top logit can cause a miss even when the model emits a valid label: scoring gives the
tie to the lowest option index, greedy `argmax` to the lowest vocabulary id. This is not corrected.

**Mass flags (mark the result, do not stop it).** Engineering defaults, not fitted to data:

| condition | action |
|---|---|
| candidate mass `M < 0.01` on an item | the item is low-mass |
| low-mass items are >= 5% of items, in either arm | flag: format mismatch |
| `M_protected < 0.1 * M_clean` on >= 5% of items | flag: mass collapse under the transform |
| free-generation agreement not measured | flag |
| non-finite mass | stop |

A flagged result has `status: "flagged"`, otherwise `"ok"`. The diagnostic is computed on the subset
(`format_report.json`) and on all scored items (`format_diagnostic` in `report.json`). Passing shows
that the label format works; it says nothing about answer quality.

## 7. Controls

Any control failure stops the run. *Extended-control items* are the first 20 items newly scored in
each phase (format validation, then the main pass) of each invocation; resumed items do not count.
The tolerance is 1e-9 absolute on every score, and the predicted label must also match. `report.json`
gives, per label, the items with saved evidence (`controls`) and this invocation's count
(`controls_this_session`). A run with no evidence for a label stops.

- **C0, clean reference.** The clean arm, scored through the same path as the protected arm. Every
  comparison is paired against it, and section 6 applies to it as to the protected arm.
- **C1, identity (every item).** A freshly built transform with the same configuration has a
  zero-initialised displacement head and must return `E` bitwise. Its bf16 cast is scored; its four
  scores and four log-probabilities must match the clean arm per item. Equal accuracy is not enough.
- **C1a, scoring oracle (extended items).** An independent uncached teacher-forced pass over ordinary
  `input_ids` must reproduce the clean scores. `C1a_multi`: four two-token continuations
  (`[label id, EOS id]`) go through the production multi-token path and the oracle, and must agree.
- **C1b, constructed fixtures (CPU).** `tests/test_mmlu_scorer.py` scores items with known logits,
  deliberate ties and every transition, and checks the bounds, the row checks and the scorer hash.
- **C1c, delivery (every item; more on extended items).** A forward pre-hook on the first decoder
  block checks, on every model call (clean, identity, protected, controls, each generated token), that
  the hidden states it receives start with the expected input tensor, bitwise. An unknown decoder
  layout stops the run. `C1c_oracle`: the protected scores must match an oracle pass that calls
  `model.forward` directly on `Z`. `C1c_nonidentity`: a synthetic prompt (`E` with 1 added to every
  component of the first position) must reach the decoder, differ from `E`, and match the oracle.
  This keeps C1c meaningful at alpha = 0.
- **C2, blend endpoints (every item; scored on extended items).** alpha = 0 must give `E` and
  alpha = 1 must give `T(E)`, bitwise in fp32. On extended items the alpha = 0 input must score as the
  clean arm and the alpha = 1 input must match the oracle on `T(E)`.
- **C3, decoder sanity (extended items).** The hiding-rate decoder on clean embeddings, in fp32 and
  bf16, must return the original token at every content position. A decoder that cannot read the
  plain input proves nothing.

C1 alone cannot see a fault both arms share, such as a wrong label mapping; C1a and C1b cover that.
C1, C1a and C1b all pass if a dispatch fault sends the clean embeddings to the model; C1c catches
that. C1c checks delivery to the first decoder block, not every later model operation.

## 8. Statistics

The unit is the item, and every comparison is paired. `n` is the number of clean-correct items and
`x` the number of breaks.

- **Upper bound (primary):** Clopper-Pearson exact one-sided 95% upper bound on `x/n`
  (`regression_upper_95`). For `x = 0` it is `1 - 0.05^(1/n)`; otherwise bisection (200 iterations)
  on `P(X <= x | n, p) = 0.05`, with the binomial tail summed in log space.
- **Lower bound:** the exact one-sided 95% lower bound (`regression_lower_95`), `1 - upper(n - x, n)`,
  and 0 when `x = 0`.
- **Subject-block bound (beside it):** with `n_s` clean-correct items in subject `s` and
  `w_s = n_s / n`, `subject_block_upper_95 = min(1, x/n + sqrt(ln(20) * sum(w_s^2) / 2))`. The root
  term is `subject_block_allowance`; `effective_subject_blocks = 1 / sum(w_s^2)`. Weights are
  clean-correct counts, not subject sizes.
- **Per subject:** items, clean-correct, protected-correct, breaks, repairs, break rate (null with no
  clean-correct item), both accuracies and the hiding rate.
- **No bootstrap:** an all-zero break indicator resamples to a false upper bound of zero.

With `n = 0` all bounds are null. Clopper-Pearson assumes independent items; the block bound assumes
independent subjects given the clean outcomes. Neither is established (section 10). When they
disagree, the block bound does not override Clopper-Pearson; both are descriptive. Bounds are per
run: comparing several alphas or checkpoints on the same items needs a multiplicity correction,
which the code does not apply.

## 9. Outputs

`python scripts/evaluate.py --checkpoint PATH [--alpha 1.0] [--limit N] [--out DIR] [--model-dir DIR]
[--data-dir DIR]` reads `data/` by default and writes to `runs/mmlu-a<alpha>/`. `--limit N` scores only the
first `N` items of `mmlu_dev.jsonl`; format validation still uses the whole subset.

- **`manifest.json`**, written before the first forward pass. `protocol_hash` is the SHA-256 of everything
  else in it: prompt text and `rendered_prefix`, label ids, alpha, seed, checkpoint SHA-256 and transform
  configuration, data file hashes, ordered item ids, a hash of the tokenized prompts, precision settings,
  `scorer_protocol_hash` (scoring code, row schema, thresholds, bound) and `system` (SHA-256 of each local
  model and tokenizer file, library versions, model and tokenizer configuration, chat template, device,
  and SHA-256 of `scripts/evaluate.py`, `scripts/common.py`, `evaluation/mmlu_scorer.py`,
  `evaluation/mmlu_contract.py`, `privacy_transform/model.py` and this file).
- **`format_rows.jsonl`** and **`rows.jsonl`**: one JSON line per item with `protocol_hash`, `result` (the
  row fields below), fp64 log-probabilities for both arms, per-label `controls` and a `row_sha` checksum.
  Format rows also carry the generated text per arm and are copied, not rescored, into `rows.jsonl`.
- **`format_report.json`**: the section 6 diagnostic on the subset, including `flags` and `status`.
- **`report.json`**: `protocol_hash`, `alpha`, `controls`, `controls_this_session`, `summary` (counts,
  rates and bounds of sections 1 and 8, `transitions`, ties, `per_subject`), `format_diagnostic`,
  `obfuscation` (the hiding rate), `diagnostics` and `claim` (scope statement).

Each row's `result` holds identity (`item_id`, `subject`, `gold`), both predictions, scores, correctness
and `transition`, candidate masses and margins, tie flags, changed-token counts (fp32 and bf16, section 5),
`content_tokens`, `prompt_tokens` and `label_changed`. Five descriptive fields enter no count, rate,
bound or gate: `clean_winner_protected_lead`, `clean_winner_lead_change`, `displacement_mean` (mean
`norm(z - e) / norm(e)` over content tokens) and `lookup_margins` with its bf16 twin (per content token,
cosine with its own row minus the best other cosine). Before a row is written or reloaded, the code
derives its flags again from scores and gold and stops on any contradiction, duplicate item id, mass
outside [0, 1 + 1e-9], invalid count or non-finite value.

**Resume.** Running again with the same `--out` resumes the run if its `manifest.json` has the same
`protocol_hash`; any other difference in the manifest, or rows without a manifest, stops the run. So a
change to the checkpoint, alpha, data, selection, model files, libraries, source files or this file starts
a new run. Each reloaded row must pass its checksum and the row checks, and match the source data. A torn
last line is dropped if its checksum fails; damage on any other line stops the run.

## 10. Limits

- **Multiple choice only.** Scoring compares option likelihoods after a fixed prefill. It does not
  test free generation, explanations or long answers; a transform can do well here and generate poorly.
- **One prompt format.** Results are conditional on the prompt of section 3. Clean accuracy here is
  not comparable to MMLU numbers from a different format, template or harness.
- **The hiding rate is not a privacy measure.** It counts the tokens one fixed nearest-neighbour
  decoder fails to read. An attacker who can study the system does much better: the subspace attack
  in `scripts/attack.py` recovers tokens that this decoder reports as hidden.
- **Questions within a subject are not independent.** They share topic, source and style, and
  subjects differ in size (5 to 170 items). Neither the item independence behind Clopper-Pearson nor
  the subject independence behind the block bound is established. No bound here supports a
  population guarantee.
- **The break rate sees only clean-correct items.** A wrong answer that becomes another wrong answer
  is not a break; the label-change rate reports it.
- **Ties.** Exact ties for the top logit can occur because the model runs in bf16. They are broken by
  option index and counted, and a tied item can miss in format validation (section 6).
- **Deduplication** uses exact normalised keys; paraphrases and translations of test items can remain.
- **The delivery check** (C1c) covers the input to the first decoder block, not later model operations.
