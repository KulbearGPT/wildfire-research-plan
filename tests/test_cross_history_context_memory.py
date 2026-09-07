import torch

from reproductions.cross_history.models import ValidContextMemoryAttention


def test_context_memory_starts_as_identity_and_never_changes_valid_pixels() -> None:
    layer = ValidContextMemoryAttention(4)
    features = torch.randn(2, 4, 8, 8)
    missing = torch.zeros(2, 1, 8, 8)
    missing[:, :, 2:6, 2:6] = 1

    initial = layer(features, missing)
    torch.testing.assert_close(initial, features)

    with torch.no_grad():
        layer.output.weight.fill_(0.1)
    changed = layer(features, missing)
    torch.testing.assert_close(changed * (1 - missing), features * (1 - missing))
    assert not torch.equal(changed * missing, features * missing)
