import math
import unittest

import torch
import torch.nn.functional as F

from reproductions.wsts_fast_track.evaluate_missingness import evaluate_batches


class Predictor(torch.nn.Module):
    def __init__(self, loss=None):
        super().__init__()
        self.loss = loss

    def forward(self, x):
        return x

    def compute_loss(self, logits, target):
        if self.loss is not None:
            return logits.new_tensor(self.loss)
        return F.binary_cross_entropy_with_logits(logits, target.float())


class UncertainPredictor(Predictor):
    def __init__(self, variance):
        super().__init__()
        self.variance = variance

    def forward_with_uncertainty(self, x):
        return x, torch.full_like(x, self.variance)


class FiniteEvaluationTests(unittest.TestCase):
    def evaluate(self, model, logits=None):
        if logits is None:
            logits = torch.tensor([[[2., -2.]]])
        target = torch.tensor([[[1, 0]]])
        return evaluate_batches(model, [(logits, target)], device=torch.device('cpu'))

    def test_rejects_nonfinite_logits(self):
        for invalid in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, 'non-finite logits'):
                    self.evaluate(Predictor(), torch.tensor([[[invalid, -2.]]]))

    def test_rejects_nonfinite_loss_with_finite_logits(self):
        for invalid in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, 'non-finite loss'):
                    self.evaluate(Predictor(loss=invalid))

    def test_rejects_nonfinite_predictive_variance(self):
        with self.assertRaisesRegex(ValueError, 'non-finite predictive variance'):
            self.evaluate(UncertainPredictor(float('nan')))

    def test_finite_predictions_preserve_metrics(self):
        metrics = self.evaluate(UncertainPredictor(.25))
        self.assertEqual(metrics['avg_precision'], 1.)
        self.assertEqual(metrics['f1'], 1.)
        self.assertEqual(metrics['pixel_count'], 2)
        self.assertEqual(metrics['mean_predictive_variance'], .25)
        self.assertAlmostEqual(metrics['loss'], math.log1p(math.exp(-2)), places=6)
        self.assertAlmostEqual(metrics['brier'], (1 / (1 + math.exp(2))) ** 2, places=6)


if __name__ == '__main__':
    unittest.main()
