import copy
import unittest

import torch
from torch import nn

from reproductions.cross_history.three_directions import (
    MomentCollector, ShallowInputStem, apply_bn_bank,
    bernoulli_teacher_kl, observable_routes,
)


class _BNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm2d(2)
        self.dropout = nn.Dropout(p=1.0)

    def forward(self, x):
        return self.dropout(self.bn2(x)), self.bn1(x.mean((-2, -1)))


class MomentCollectorTests(unittest.TestCase):
    def test_collects_unbiased_common_and_independent_bank_moments(self):
        model = _BNModel().train()
        before = copy.deepcopy(model.state_dict())
        collector = MomentCollector(model)
        collector.bank = 0
        out0, _ = model(torch.tensor([[[[1.]], [[3.]]], [[[3.]], [[5.]]]]))
        collector.bank = 1
        out1, _ = model(torch.tensor([[[[7.]], [[9.]]], [[[9.]], [[11.]]]]))
        common, banks = collector.finalize()
        self.assertFalse(model.training)
        self.assertTrue(torch.allclose(out0, torch.tensor([[[[1.]], [[3.]]], [[[3.]], [[5.]]]]), atol=1e-4))
        self.assertTrue(torch.allclose(out1, torch.tensor([[[[7.]], [[9.]]], [[[9.]], [[11.]]]]), atol=1e-4))
        self.assertTrue(torch.allclose(common["bn2"]["mean"], torch.tensor([5., 7.])))
        self.assertTrue(torch.allclose(common["bn2"]["var"], torch.tensor([40 / 3, 40 / 3])))
        self.assertTrue(torch.allclose(banks[0]["bn1"]["mean"], torch.tensor([2., 4.])))
        self.assertTrue(torch.allclose(banks[0]["bn1"]["var"], torch.tensor([2., 2.])))
        self.assertTrue(torch.allclose(banks[1]["bn2"]["mean"], torch.tensor([8., 10.])))
        self.assertEqual(common["bn2"]["count"], 4)
        for name, value in before.items():
            self.assertTrue(torch.equal(model.state_dict()[name], value), name)

    def test_applies_named_bank_with_shape_checks_and_preserves_affine(self):
        model = _BNModel()
        weight = model.bn2.weight.detach().clone()
        bank = {
            "bn1": {"mean": torch.tensor([1., 2.]), "var": torch.tensor([3., 4.]), "count": 2},
            "bn2": {"mean": torch.tensor([5., 6.]), "var": torch.tensor([7., 8.]), "count": 2},
        }
        apply_bn_bank(model, bank)
        self.assertTrue(torch.equal(model.bn2.running_mean, torch.tensor([5., 6.])))
        self.assertTrue(torch.equal(model.bn2.running_var, torch.tensor([7., 8.])))
        self.assertTrue(torch.equal(model.bn2.weight, weight))
        reloaded = _BNModel()
        apply_bn_bank(reloaded, bank)
        self.assertTrue(torch.equal(reloaded.bn1.running_mean, torch.tensor([1., 2.])))
        malformed = dict(bank)
        malformed["bn2"] = {"mean": torch.ones(3), "var": torch.ones(3), "count": 2}
        with self.assertRaises(ValueError):
            apply_bn_bank(model, malformed)

    def test_empty_bank_is_rejected(self):
        model = _BNModel()
        collector = MomentCollector(model)
        collector.bank = 0
        model(torch.ones(2, 2, 1, 1))
        with self.assertRaises(ValueError):
            collector.finalize()


class ShallowInputStemTests(unittest.TestCase):
    def test_modes_are_identity_and_have_matching_parameter_counts(self):
        x = torch.randn(2, 3, 4, 5, 5)
        mixed = ShallowInputStem(4, (0, 2), "mixed")
        typed = ShallowInputStem(4, (0, 2), "typed")
        self.assertTrue(torch.equal(mixed(x), x))
        self.assertTrue(torch.equal(typed(x), x))
        self.assertEqual(sum(p.numel() for p in mixed.parameters()), 1152)
        self.assertEqual(sum(p.numel() for p in typed.parameters()), 1152)

    def test_typed_paths_keep_groups_separate_and_receive_gradients(self):
        stem = ShallowInputStem(4, (0, 2), "typed")
        for path in stem.paths:
            nn.init.constant_(path[-1].weight, 0.1)
        x = torch.randn(2, 4, 5, 5, requires_grad=True)
        changed = x.detach().clone()
        changed[:, (1, 3)] += 4
        y = stem(x)
        y_changed = stem(changed)
        self.assertTrue(torch.allclose(y[:, (0, 2)], y_changed[:, (0, 2)]))
        y.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(all(p.grad is not None for p in stem.parameters()))

    def test_rejects_invalid_groups(self):
        with self.assertRaises(ValueError):
            ShallowInputStem(3, (), "typed")
        with self.assertRaises(ValueError):
            ShallowInputStem(3, (0, 3), "typed")
        with self.assertRaises(ValueError):
            ShallowInputStem(3, (0,), "unknown")


class ObservableRoutesTests(unittest.TestCase):
    def test_routes_use_only_masks_with_block_precedence(self):
        packed = torch.zeros(4, 2, 6, 2, 2)
        packed[1, :, -1] = 1
        packed[2, :, -2, 0, 0] = 1
        packed[2, :, -1] = 1
        packed[3, :, -2, :, 0] = 1
        self.assertTrue(torch.equal(observable_routes(packed), torch.tensor([0, 1, 2, 3])))


class BernoulliTeacherKLTests(unittest.TestCase):
    def test_identity_nonnegative_extreme_finite_and_gradient_ownership(self):
        student = torch.tensor([-100., -1., 0., 2., 100.], requires_grad=True)
        teacher = student.detach().clone().requires_grad_(True)
        identity = bernoulli_teacher_kl(student, teacher)
        self.assertTrue(torch.isfinite(identity))
        self.assertGreaterEqual(identity.item(), -1e-7)
        self.assertAlmostEqual(identity.item(), 0.0, places=6)
        loss = bernoulli_teacher_kl(student, torch.zeros_like(teacher))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_nonfinite_logits_are_rejected(self):
        with self.assertRaises(ValueError):
            bernoulli_teacher_kl(torch.tensor([float("inf")]), torch.zeros(1))


if __name__ == "__main__":
    unittest.main()
