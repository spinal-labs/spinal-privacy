# spinal-privacy

Code for *Privacy-preserving inference on hosted models: a learned input-embedding transform for frozen
Qwen3.8-27B*.

A language model turns text into vectors once, with a public lookup table. This code does that step on the
client and passes the vectors through a small trained network (55.6M parameters) before they leave. The host
runs a frozen Qwen3.8-27B on the transformed vectors and never receives text or token ids.

## How it works

```
CLIENT (trusted)                                   HOST (untrusted)
prompt -> tokenise -> embedding lookup E           frozen Qwen3.8-27B
       -> transform Z = E + a·Δ(E)  ───── Z ─────>  (receives a bf16 tensor, no tokens)
                                                    -> answer (not protected)
```

- `Δ(E)` comes from a 3-layer contextual network with a 1024-wide bottleneck, so each displacement depends on
  the whole prompt.
- The transform starts as the exact identity. It is trained against the frozen model with a forward-KL
  utility term and a term that pushes each vector away from its original direction.
- `a` in [0, 1] sets the strength at inference time. Positions of the tokenizer's registered special tokens
  pass through unchanged; every other position, including the chat template's role words, is transformed.

## Results

All numbers come from one checkpoint, scored on 1,490 MMLU questions that were held out from training
(strength `a = 1`, noise off). The files in `results/` hold the numbers, and
`results/main_result_per_question.jsonl` holds the per-question rows, so you can recount them without a GPU.

| | Without transform | With transform |
|---|---:|---:|
| Correct answers | 1,226 | 1,203 |
| Accuracy | 82.3% | 80.7% |
| Right → wrong (breaks) | | 68 (5.55% of 1,226; one-sided 95% bounds 4.51% and 6.74%) |
| Wrong → right (repairs, never offset breaks) | | 45 |
| Content tokens hidden from nearest-neighbour lookup (FP32 transform output) | 0% | 30.08% (macro), 24.85% (pooled) |

- **Strength sweep** (`results/strength_sweep.json`): lowering `a` does not help. At `a = 0.8` only 1.1% of
  tokens are hidden (FP32 transform output) while 48 answers still break.
- **Chosen-input attack** (`results/attack_result.json`): an attacker who can run the transform on its own
  prompts estimates the displacement subspace (rank 64) and projects it out. This recovers **2,794 of 2,794**
  content tokens on 96 held-out Alpaca prompts (measured on the FP32 transform outputs, before the bf16
  conversion for delivery). The hiding rate does not measure this attack. The file holds the two
  nearest-neighbour baselines and the rank-64 attack.

Files in `results/`:

- `main_result.json`: the table above.
- `main_result_per_question.jsonl`: one row per question, for recounting.
- `strength_sweep.json`: breaks and hiding rate at each strength `a`.
- `attack_result.json`: the two baselines and the rank-64 chosen-input attack.

## Setup

Python 3.10 or newer. The reported runs used Python 3.12, CUDA 13.0 and one NVIDIA H200.

```bash
git clone https://github.com/spinal-labs/spinal-privacy && cd spinal-privacy
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest -q                                   # CPU only, a few seconds
```

Training, scoring and the attack need a CUDA GPU with enough memory for Qwen3.8-27B in bf16, and the model
at the pinned revision:

```bash
hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
    --local-dir models/Qwen3.8-27B
# or set QWEN_MODEL_DIR to an existing copy at that revision
```

## Usage

```bash
# Train: writes fixed_fp64.pt, rows-fixed_fp64.json and report.json to runs/train
python scripts/train.py

# Score MMLU at strength a = 1.0: writes to runs/mmlu-a1.0
python scripts/evaluate.py --checkpoint runs/train/fixed_fp64.pt --alpha 1.0

# Run the chosen-input attack: writes to runs/attack
python scripts/attack.py --checkpoint runs/train/fixed_fp64.pt
```

Options (see `--help` on each script):

- `--model-dir DIR` (all scripts): the model folder, if not `models/Qwen3.8-27B` or `$QWEN_MODEL_DIR`.
- `--out DIR` (all scripts): the output folder.
- `--overwrite` (`train.py`, `attack.py`): write into a non-empty output folder. Without it they refuse.
- `--data-dir DIR`, `--limit N` (`evaluate.py`): the data folder, and a short run on the first N questions.

`evaluate.py` resumes an output folder that already holds a run with the same manifest. See
[PROTOCOL.md](PROTOCOL.md) for the scoring rules.

To reproduce the strength sweep:

```bash
for a in 0.0 0.5 0.75 0.8 0.85 0.9 0.95 1.0; do
    python scripts/evaluate.py --checkpoint runs/train/fixed_fp64.pt --alpha $a
done
```

The files in `results/` are exports of the original run outputs, limited to the reported numbers and the
per-question rows. The raw run manifests are not included because they record machine-specific paths.

## Data

- `data/alpaca.jsonl`: 1,019 examples from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
  (CC BY-NC 4.0), unchanged except that the input is joined to the instruction.
- `data/mmlu_*.jsonl`: questions from the [MMLU](https://huggingface.co/datasets/cais/mmlu) validation split
  (MIT). `mmlu_dev.jsonl` is the 1,490-question scoring split: the 1,531 validation items minus every item
  that overlaps the upstream test split or repeats within validation.

## Citation

See [CITATION.cff](CITATION.cff).

## License

See [LICENSE](LICENSE).
