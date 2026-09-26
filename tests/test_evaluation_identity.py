"""Checkpoint identity checks using metadata only; no model/tensor imports."""
import unittest

from reproductions.cross_history.checkpoint_contract import require_evaluation_identity


class EvaluationIdentityTests(unittest.TestCase):
    def check(self, payload, **overrides):
        requested = dict(history=1, method='block_specialist', seed=0, block_fraction=.25)
        requested.update(overrides)
        require_evaluation_identity(payload, **requested)

    def test_matching_roles_and_both_histories(self):
        for h in (1, 5):
            for seed in (0, 1, 2):
                for method, fraction in (('control', None), ('cosine_erm', None),
                                         ('block_specialist', None),
                                         ('block_specialist', .25), ('block_specialist', .5)):
                    payload = dict(history=h, seed=seed, method=method, block_fraction=fraction)
                    self.check(payload, **payload)

    def test_different_seed_or_expert_cannot_be_relabelled(self):
        base = dict(history=1, seed=0, method='block_specialist', block_fraction=.25)
        for key, value in [('history', 5), ('history', True), ('method', 'cosine_erm'),
                           ('seed', 1), ('seed', False), ('block_fraction', .5),
                           ('block_fraction', None)]:
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, key):
                    self.check(dict(base, **{key: value}))

    def test_missing_identity_rejected_except_legacy_mixed_fraction(self):
        payload = dict(history=1, seed=0, method='block_specialist')
        self.check(payload, block_fraction=None)
        with self.assertRaisesRegex(ValueError, 'block_fraction'):
            self.check(payload)
        for key in ('history', 'method', 'seed'):
            incomplete = dict(payload)
            incomplete.pop(key)
            with self.assertRaisesRegex(ValueError, key):
                self.check(incomplete, block_fraction=None)

    def test_archived_control_transform_remains_explicit(self):
        payload = dict(history=1, seed=0, method='control')
        self.check(payload, method='normalized_inpaint', block_fraction=None)
        with self.assertRaisesRegex(ValueError, 'seed'):
            self.check(payload, method='normalized_inpaint', seed=1, block_fraction=None)
        with self.assertRaisesRegex(ValueError, 'method'):
            self.check(dict(payload, method='cosine_erm'),
                       method='normalized_inpaint', block_fraction=None)


if __name__ == '__main__':
    unittest.main()
