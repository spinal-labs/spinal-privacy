"""Train the input-embedding transform against the frozen Qwen3.8-27B target.

The distillation loss is computed in FP64 (the loss default). A real answer mask keeps answer padding
out of the loss, so the loss is not fitted to artificial repeated EOS tokens. Every batch holds prompts
of exactly equal length, so no row takes its first answer prediction from a padding position.

Training starts from the identity transform, which is checked before the first step. All 96 held-out
prompts are evaluated and their per-prompt rows are saved, so uncertainty can be computed from them.

Usage:  python scripts/train.py [--model-dir DIR] [--out DIR] [--overwrite]
Writes: OUT/fixed_fp64.pt (checkpoint), OUT/rows-fixed_fp64.json, OUT/report.json
"""

import argparse
import copy
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch  # noqa: E402

import common  # noqa: E402
from privacy_transform.model import PrivacyTransform, TransformConfig  # noqa: E402
from training.trainer import Trainer, TrainingConfig  # noqa: E402

OUT = REPO / 'runs' / 'train'
SEED = 20260909
ARM = 'fixed_fp64'  # names the checkpoint and the rows file
COSINE = 0.3
TOKEN_BUDGET = 512
TARGET_EXAMPLES = 20000
MAXLEN = 96
EVALS = 3
REPORT = {
    'status': 'running',
    'scope': '27B, FP64 distillation loss, answer mask, equal-length prompt batches',
    'cosine_weight': COSINE,
    'target_examples': TARGET_EXAMPLES,
    'seed': SEED,
}

Record = dict[str, Any]


def buckets(records: list[Record], budget: int) -> list[list[Record]]:
    """Batches of EXACTLY equal prompt length and at most budget prompt tokens, so no row predicts
    from padding."""
    by_length = defaultdict(list)
    for record in records:
        by_length[len(record['prompt_ids'])].append(record)
    batches = []
    for length, group in sorted(by_length.items()):
        size = max(1, budget // length)
        batches += [group[i : i + size] for i in range(0, len(group), size)]
    return batches


def main(args: argparse.Namespace) -> None:
    """Train from the identity transform and evaluate the held-out prompts along the way."""
    torch.backends.cuda.matmul.allow_tf32 = False
    common.seed(SEED)
    tok, model = common.load(args.model_dir)
    embedding = model.get_input_embeddings()
    table = embedding.weight.detach()
    eos = {i for i in (model.generation_config.eos_token_id or []) if i is not None} or {tok.eos_token_id}

    records = [record for record in common.corpus(tok) if len(record['prompt_ids']) <= MAXLEN]
    for record in records:
        answer_ids = tok(record['reference'], add_special_tokens=False)['input_ids'][:32]
        record['answer_ids'] = answer_ids or [tok.eos_token_id]
    train, test = records[:800], records[800:896]
    plan = buckets(train, TOKEN_BUDGET)
    mean_batch = sum(len(batch) for batch in plan) / len(plan)
    lr = 3e-5 * (4.0 / mean_batch) ** 0.5
    REPORT['batching'] = {'batches': len(plan), 'mean_batch': mean_batch, 'lr': lr}
    common.event('batching', **REPORT['batching'])

    # The frozen model's clean outputs never change, so they are generated once.
    clean = {}
    for record in test:
        prompt = embedding(torch.tensor(record['prompt_ids'], device='cuda'))[None]
        clean[record['id']] = common.generate(model, prompt, eos, limit=64)
    common.event('clean_cached', prompts=len(clean))

    def batch_tensors(group: list[Record]) -> tuple[torch.Tensor, ...]:
        """Embeddings, ids, attention mask, content mask, answer ids and answer mask of one batch."""
        width = max(len(record['prompt_ids']) for record in group)
        answer_width = max(len(record['answer_ids']) for record in group)
        ids = torch.zeros(len(group), width, dtype=torch.long, device='cuda')
        attention = torch.zeros(len(group), width, dtype=torch.bool, device='cuda')
        content = torch.zeros(len(group), width, dtype=torch.bool, device='cuda')
        answers = torch.full((len(group), answer_width), tok.eos_token_id, dtype=torch.long, device='cuda')
        answer_mask = torch.zeros(len(group), answer_width, dtype=torch.bool, device='cuda')
        for i, record in enumerate(group):
            n, a = len(record['prompt_ids']), len(record['answer_ids'])
            ids[i, :n] = torch.tensor(record['prompt_ids'], device='cuda')
            attention[i, :n] = True
            content[i, :n] = torch.tensor(record['content_mask'], device='cuda')
            answers[i, :a] = torch.tensor(record['answer_ids'], device='cuda')
            answer_mask[i, :a] = True  # padding is excluded from the loss and from attention
        return table[ids].float(), ids, attention, content, answers, answer_mask

    config = TransformConfig(embedding_dim=5120, hidden_dim=1024, sampling_space='output')
    config.validate_target()
    common.seed(SEED)
    fresh = PrivacyTransform(config).cuda()
    initial = copy.deepcopy(fresh.state_dict())
    del fresh
    gc.collect()
    torch.cuda.empty_cache()

    def evaluate(transform: PrivacyTransform, label: str, seen: int) -> tuple[dict[str, Any], list[Record]]:
        """All 96 held-out prompts, noise off, per-prompt rows retained."""
        transform.eval().requires_grad_(False)
        rows = []
        with torch.no_grad():
            for record in test:
                e = embedding(torch.tensor(record['prompt_ids'], device='cuda'))[None].float()
                attention = torch.ones(e.shape[:2], dtype=torch.bool, device='cuda')
                content = torch.tensor(record['content_mask'], device='cuda')[None]
                z = transform(e, attention, content, stochastic=False)[0]
                predicted = common.nearest(z, table).tolist()
                changed = sum(
                    got != want
                    for got, want, keep in zip(predicted, record['prompt_ids'], record['content_mask'])
                    if keep
                )
                total = sum(record['content_mask'])
                output = common.generate(model, z[None], eos, limit=64)
                match = common.agreement(clean[record['id']], output)
                rows.append(
                    {
                        'id': record['id'],
                        'prompt_tokens': len(record['prompt_ids']),
                        'tokens_changed_pct': 100.0 * changed / max(total, 1),
                        **{k: v for k, v in match.items() if k != 'token_agreement'},
                        'token_agreement': 100.0 * match['token_agreement'],
                    }
                )
        transform.train().requires_grad_(True)
        n = len(rows)

        def mean(key: str) -> float:
            return sum(row[key] for row in rows) / n

        def sd(key: str) -> float:
            return (sum((row[key] - mean(key)) ** 2 for row in rows) / max(n - 1, 1)) ** 0.5

        point = {
            'label': label,
            'examples_seen': seen,
            'n': n,
            'tokens_changed_pct': mean('tokens_changed_pct'),
            'token_agreement': mean('token_agreement'),
            'agreement_sd': sd('token_agreement'),
            'agreement_stderr': sd('token_agreement') / n**0.5,
            'exact_pct': 100.0 * sum(row['exact'] for row in rows) / n,
            'exact_count': sum(row['exact'] for row in rows),
            **common.divergence_summary(rows),
        }
        common.event('evaluated', **point)
        return point, rows

    started = time.monotonic()
    common.seed(SEED)
    transform = PrivacyTransform(config).cuda()
    transform.load_state_dict({k: v.cuda() for k, v in initial.items()})
    base, base_rows = evaluate(transform, f'{ARM}@identity', 0)
    assert base['tokens_changed_pct'] == 0.0 and base['token_agreement'] == 100.0, 'identity gate failed'
    trainer = Trainer(
        model,
        transform,
        config=TrainingConfig(
            learning_rate=lr,
            utility_weight=1.0,
            cosine_weight=COSINE,
            norm_weight=0.1,
            variance_weight=0.001,
        ),
    )
    seen, step, marks, order = 0, 0, [], list(range(len(plan)))
    every = TARGET_EXAMPLES // EVALS
    while seen < TARGET_EXAMPLES:
        random.shuffle(order)
        for i in order:
            if seen >= TARGET_EXAMPLES:
                break
            embeddings, ids, attention, content, answers, answer_mask = batch_tensors(plan[i])
            trainer.step(embeddings, ids, attention, content, answers, answer_mask=answer_mask)
            seen += len(plan[i])
            step += 1
            if seen // every > len(marks) and seen < TARGET_EXAMPLES:
                marks.append(evaluate(transform, f'{ARM}@{seen}', seen)[0])
    final, final_rows = evaluate(transform, f'{ARM}-final', seen)
    rows_text = json.dumps({'identity': base_rows, 'final': final_rows}, indent=2) + '\n'
    (args.out / f'rows-{ARM}.json').write_text(rows_text)
    torch.save(
        {
            'config': transform.config_dict(),
            'arm': ARM,
            'state_dict': {k: v.detach().cpu() for k, v in transform.state_dict().items()},
        },
        args.out / f'{ARM}.pt',
    )
    REPORT['arms'] = [
        {
            'arm': ARM,
            'identity': base,
            'trajectory': marks + [final],
            'steps': step,
            'seconds': time.monotonic() - started,
        }
    ]
    REPORT['status'] = 'completed_single_arm'
    common.event('completed_single_arm', arm=ARM, final=final)


def parse_args() -> argparse.Namespace:
    """Parse the command line and refuse a non-empty output folder unless --overwrite is given."""
    parser = argparse.ArgumentParser(
        description='Train the input-embedding transform (needs a CUDA GPU and the model).'
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        default=common.MODEL,
        help='local Qwen3.8-27B folder (default: $QWEN_MODEL_DIR or models/Qwen3.8-27B)',
    )
    parser.add_argument('--out', type=Path, default=OUT, help='output folder (default: runs/train)')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='write into a non-empty output folder, replacing files with the same names',
    )
    args = parser.parse_args()
    common.refuse_nonempty_output(parser, args.out, args.overwrite)
    return args


if __name__ == '__main__':
    sys.exit(common.run_with_report(main, parse_args(), REPORT))
