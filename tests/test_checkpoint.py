"""A .pt checkpoint and its model.safetensors + config.json copy load to identical transforms."""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import common  # noqa: E402
from privacy_transform.model import PrivacyTransform, TransformConfig  # noqa: E402


def _trained_transform() -> PrivacyTransform:
    torch.manual_seed(0)
    config = TransformConfig(embedding_dim=16, hidden_dim=8, num_layers=2, num_heads=2, feedforward_dim=16)
    transform = PrivacyTransform(config)
    with torch.no_grad():
        for parameter in transform.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    return transform.eval()


def test_pt_and_safetensors_checkpoints_load_identically(tmp_path: Path) -> None:
    transform = _trained_transform()
    config = transform.config_dict()
    state_dict = {k: v.detach().clone() for k, v in transform.state_dict().items()}

    pt_file = tmp_path / 'transform.pt'
    torch.save({'config': config, 'state_dict': state_dict}, pt_file)
    folder = tmp_path / 'published'
    folder.mkdir()
    save_file(state_dict, str(folder / 'model.safetensors'))
    (folder / 'config.json').write_text(json.dumps(config))

    from_pt = common.load_checkpoint(pt_file)
    from_folder = common.load_checkpoint(folder)
    from_file = common.load_checkpoint(folder / 'model.safetensors')

    assert from_pt['weights_file'] == pt_file
    assert from_folder['weights_file'] == from_file['weights_file'] == folder / 'model.safetensors'
    for loaded in (from_folder, from_file):
        assert loaded['config'] == from_pt['config'] == config
        assert loaded['state_dict'].keys() == from_pt['state_dict'].keys()
        for key, tensor in from_pt['state_dict'].items():
            assert torch.equal(loaded['state_dict'][key], tensor)

    rebuilt = PrivacyTransform(TransformConfig(**from_folder['config'])).eval()
    rebuilt.load_state_dict(from_folder['state_dict'])
    embeddings = torch.randn(2, 5, 16)
    attention = torch.ones(2, 5, dtype=torch.bool)
    content = torch.tensor([[True, True, False, True, True], [True, False, True, True, True]])
    with torch.no_grad():
        expected = transform(embeddings, attention, content, stochastic=False)
        assert torch.equal(rebuilt(embeddings, attention, content, stochastic=False), expected)
