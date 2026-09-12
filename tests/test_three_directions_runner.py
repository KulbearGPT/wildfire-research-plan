import importlib
import importlib.util
import unittest


class RunnerContractTests(unittest.TestCase):
    def module(self):
        name = 'reproductions.cross_history.run_three_directions'
        self.assertIsNotNone(importlib.util.find_spec(name), 'experiment driver missing')
        return importlib.import_module(name)

    def test_checkpoint_contract_rejects_wrong_teacher(self):
        validate = self.module().validate_source
        record = dict(history=5, seed=0, architecture='res18_utae',
                      method='block_specialist', block_fraction=.25)
        payload = dict(record)
        validate(payload, record)
        for key, bad in [('history', 1), ('seed', 1),
                         ('architecture', 'swin_unet'), ('method', 'control'),
                         ('block_fraction', .5)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate({**payload, key: bad}, record)

    def test_stem_is_applied_before_original_feature_mixing(self):
        import torch
        from torch import nn
        module = self.module()
        class Recorder(nn.Module):
            def __init__(self):
                super().__init__()
                self.channels = 2
                self.last = None
            def forward(self, x):
                self.last = x.clone()
                return x[:, 0, :1]
            def compute_loss(self, x, y):
                return x.sum()
        class PlusOne(nn.Module):
            def forward(self, x):
                return x + 1
        base = Recorder()
        wrapped = module.StemForecaster(base, PlusOne())
        packed = torch.zeros(2, 5, 4, 3, 3)
        packed[:, :, -2:] = .25
        result = wrapped(packed)
        torch.testing.assert_close(base.last[:, :, :2], torch.ones(2,5,2,3,3))
        torch.testing.assert_close(base.last[:, :, -2:], packed[:, :, -2:])
        self.assertEqual(result.shape, (2,1,3,3))

    def test_routed_teacher_receives_same_corrupt_input_and_keeps_order(self):
        import torch
        from torch import nn
        module = self.module()
        class Teacher(nn.Module):
            def __init__(self, offset):
                super().__init__()
                self.offset = offset
            def forward(self, x):
                return x[:,0,:1] + self.offset
            def compute_loss(self, x, y):
                return x.sum()
        model = module.RoutedTeacher([Teacher(i) for i in range(4)])
        packed = torch.zeros(4,1,4,4,4)
        packed[:,0,0] = torch.arange(4)[:,None,None]*10
        packed[0,0,-2,:2] = 1
        packed[1,0,-1] = 1
        packed[2,0,-2,:1] = 1
        out = model(packed)
        torch.testing.assert_close(out[:,0,0,0], torch.tensor([3.,11.,22.,30.]))


if __name__ == '__main__':
    unittest.main()
