import torch

from reproductions.cross_history.models import ChangeAuxiliaryHead


def test_change_auxiliary_head_starts_from_main_forecast() -> None:
    head = ChangeAuxiliaryHead(6)
    decoded = torch.randn(2, 6, 9, 11)
    main_logits = torch.randn(2, 1, 9, 11)

    state, survival, new = head(decoded, main_logits)

    torch.testing.assert_close(state, torch.zeros_like(state))
    torch.testing.assert_close(survival, main_logits)
    torch.testing.assert_close(new, main_logits)
