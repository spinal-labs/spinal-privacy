"""Chosen-input subspace attack on a trained transform checkpoint.

The attacker submits its own prompts, knows their embeddings E exactly and sees the transformed
embeddings Z. An SVD of the content-token deltas Z - E estimates the displacement subspace. Projecting
that subspace (and the mean shift) out of the victim's Z and out of the vocabulary table, then decoding
by nearest neighbour, recovers the victim's tokens. No transform weights are needed.

Two baselines are reported with the attack: nearest-neighbour decoding of the raw embeddings (a positive
control: it should recover every token, so a lower score points to a decoding problem) and of the
unprojected Z.

Usage:  python scripts/attack.py --checkpoint runs/train/fixed_fp64.pt [--model-dir DIR] [--out DIR]
                                 [--overwrite]
Writes: OUT/report.json
"""

import argparse
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

import common  # noqa: E402
from privacy_transform.model import PrivacyTransform, TransformConfig  # noqa: E402

OUT = REPO / 'runs' / 'attack'
SEED = 20260909
RANK = 64
MAXLEN = 96
REPORT = {
    'status': 'running',
    'scope': '27B chosen-input attack',
    'privacy_certified': False,
    'rank': RANK,
    'seed': SEED,
}


@torch.no_grad()
def project_nearest(
    observed: torch.Tensor,
    table: torch.Tensor,
    basis: torch.Tensor | None,
    shift: torch.Tensor,
    chunk: int = 8192,
) -> torch.Tensor:
    """Subtract shift, remove span(basis) from both sides, then decode by cosine nearest neighbour.

    basis is [dim, k] with orthonormal columns, or None for plain nearest-neighbour decoding.
    """
    z = observed.float() - shift
    t = table.float()
    if basis is not None:
        z = z - (z @ basis) @ basis.T
        t = t - (t @ basis) @ basis.T
    z = F.normalize(z, dim=-1)
    best = z.new_full((len(z),), -float('inf'))
    ids = torch.zeros(len(z), dtype=torch.long, device=z.device)
    for off in range(0, len(t), chunk):
        cand = F.normalize(t[off : off + chunk], dim=-1)
        score, local = (z @ cand.T).max(-1)
        ids = torch.where(score > best, local + off, ids)
        best = torch.maximum(score, best)
    return ids


@torch.no_grad()
def displacement_subspace(deltas: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Estimate the displacement subspace from the attacker's deltas Z - E, one row per token.

    Returns the [dim, rank] orthonormal basis of the top right singular vectors of the centred deltas,
    the mean shift, and the share of the centred delta energy that the basis captures.
    """
    shift = deltas.mean(0)
    _, sv, right = torch.linalg.svd(deltas - shift, full_matrices=False)
    energy = float((sv[:rank] ** 2).sum() / (sv**2).sum())
    return right[:rank].T.contiguous(), shift, energy


def main(args: argparse.Namespace) -> None:
    """Report the two nearest-neighbour baselines and the rank-64 chosen-input attack."""
    torch.backends.cuda.matmul.allow_tf32 = False
    common.seed(SEED)
    tok, model = common.load(args.model_dir)
    embedding = model.get_input_embeddings()
    table = embedding.weight.detach()

    records = [record for record in common.corpus(tok) if len(record['prompt_ids']) <= MAXLEN]
    attacker_prompts, victim_prompts = records[:200], records[800:896]
    assert not ({r['id'] for r in attacker_prompts} & {r['id'] for r in victim_prompts})
    REPORT['sets'] = {'attacker_prompts': len(attacker_prompts), 'victim_prompts': len(victim_prompts)}
    common.event('sets', **REPORT['sets'])

    state = common.load_checkpoint(args.checkpoint)
    transform = PrivacyTransform(TransformConfig(**state['config'])).cuda().eval().requires_grad_(False)
    transform.load_state_dict(state['state_dict'])

    @torch.no_grad()
    def protect(record: dict[str, Any]) -> torch.Tensor:
        """Transformed embeddings Z of one prompt, noise off."""
        e = embedding(torch.tensor(record['prompt_ids'], device='cuda'))[None].float()
        attention = torch.ones(e.shape[:2], dtype=torch.bool, device='cuda')
        content = torch.tensor(record['content_mask'], device='cuda')[None]
        return transform(e, attention, content, stochastic=False)[0]

    with torch.no_grad():
        observed = torch.cat([protect(record) for record in victim_prompts])
        raw = torch.cat(
            [
                embedding(torch.tensor(record['prompt_ids'], device='cuda')).float()
                for record in victim_prompts
            ]
        )
        keep = torch.cat([torch.tensor(record['content_mask'], device='cuda') for record in victim_prompts])

    def score(attack: str, predicted: torch.Tensor) -> dict[str, Any]:
        """Count the victim's content tokens that predicted recovers."""
        truth = torch.cat([torch.tensor(record['prompt_ids'], device='cuda') for record in victim_prompts])
        correct = int((predicted[keep] == truth[keep]).sum())
        total = int(keep.sum())
        row = {
            'attack': attack,
            'content_tokens_recovered': correct,
            'content_tokens_total': total,
            'recovery_pct': 100.0 * correct / total,
        }
        common.event('attack', **row)
        return row

    zero = torch.zeros(state['config']['embedding_dim'], device='cuda')
    rows = [
        score('raw_positive_control', project_nearest(raw, table, None, zero)),
        score('unprojected_nearest', project_nearest(observed, table, None, zero)),
    ]

    # Chosen input: exact deltas from the attacker's own prompts, content tokens only.
    with torch.no_grad():
        pieces = []
        for record in attacker_prompts:
            z = protect(record)
            e = embedding(torch.tensor(record['prompt_ids'], device='cuda')).float()
            content = torch.tensor(record['content_mask'], device='cuda')
            pieces.append((z - e)[content])
        deltas = torch.cat(pieces)
    basis, shift, energy = displacement_subspace(deltas, RANK)
    energy_key = f'displacement_energy_captured_at_rank_{RANK}'
    common.event('delta_spectrum', vectors=int(len(deltas)), **{energy_key: energy})
    rows.append(score(f'chosen_plaintext_rank_{RANK}', project_nearest(observed, table, basis, shift)))

    REPORT['checkpoint'] = str(args.checkpoint)
    REPORT['rows'] = rows
    REPORT[energy_key] = energy
    REPORT['best_attack_pct'] = max(
        row['recovery_pct'] for row in rows if row['attack'] != 'raw_positive_control'
    )
    REPORT['status'] = 'completed'
    common.event('completed', best_attack_pct=REPORT['best_attack_pct'])


def parse_args() -> argparse.Namespace:
    """Parse the command line and refuse a non-empty output folder unless --overwrite is given."""
    parser = argparse.ArgumentParser(
        description='Chosen-input subspace attack on a transform checkpoint (needs a CUDA GPU and the model).'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='transform checkpoint: a .pt file from scripts/train.py, or a folder with model.safetensors '
        'and config.json',
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        default=common.MODEL,
        help='local Qwen3.8-27B folder (default: $QWEN_MODEL_DIR or models/Qwen3.8-27B)',
    )
    parser.add_argument('--out', type=Path, default=OUT, help='output folder (default: runs/attack)')
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
