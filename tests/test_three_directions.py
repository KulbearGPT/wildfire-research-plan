import unittest
import torch
from torch import nn
from reproductions.three_directions.methods import (
    StatsBatchNorm, StatisticsModel, condition_ids, midpoint_state,
    bernoulli_kl, teacher_ids,
)


class MethodsTests(unittest.TestCase):
    def test_four_conditions_and_teacher_routes(self):
        x=torch.zeros(6,1,3,4,4)
        x[1,:,-1]=1
        x[2,:,-2:,0,:]=1
        x[3,:,-2:,:2,:]=1
        x[4,:,-2:,0,:]=1; x[4,:,-1]=1
        x[5,:,-2:,:2,:]=1; x[5,:,-1]=1
        self.assertEqual(condition_ids(x).tolist(),[0,1,2,2,3,3])
        self.assertEqual(teacher_ids(x).tolist(),[0,1,2,3,1,1])

    def test_initial_bn_equivalence_and_affine_preservation(self):
        old=nn.BatchNorm2d(2).eval()
        old.running_mean.copy_(torch.tensor([1.,2.])); old.running_var.fill_(3)
        old.weight.data.fill_(2); old.bias.data.fill_(.5)
        new=StatsBatchNorm(old,4)
        x=torch.randn(4,2,3,3); groups=torch.arange(4)
        torch.testing.assert_close(new(x,groups),old(x))
        new.begin_calibration(); new(x,groups); new.finish_calibration()
        torch.testing.assert_close(new.weight,old.weight)
        torch.testing.assert_close(new.bias,old.bias)

    def test_calibration_pools_exact_pixel_moments(self):
        bn=StatsBatchNorm(nn.BatchNorm2d(1),2)
        bn.begin_calibration()
        a=torch.tensor([[[[1.,3.]]],[[[10.,14.]]]])
        bn(a,torch.tensor([0,1])); bn(a+2,torch.tensor([0,1]))
        bn.finish_calibration()
        torch.testing.assert_close(bn.running_mean[:,0],torch.tensor([3.,13.]))
        torch.testing.assert_close(bn.running_var[:,0],torch.tensor([8/3,20/3]))
        self.assertEqual(bn.count.tolist(),[4,4])

    def test_empty_group_is_rejected(self):
        bn=StatsBatchNorm(nn.BatchNorm2d(1),2); bn.begin_calibration()
        bn(torch.ones(1,1,2,2),torch.tensor([0]))
        with self.assertRaisesRegex(ValueError,'support'):bn.finish_calibration()

    def test_midpoint_and_incompatible_buffers(self):
        left={'weight':torch.tensor([1.,3.]),'bn.num_batches_tracked':torch.tensor(4)}
        right={'weight':torch.tensor([3.,5.]),'bn.num_batches_tracked':torch.tensor(9)}
        torch.testing.assert_close(midpoint_state(left,right)['weight'],torch.tensor([2.,4.]))
        with self.assertRaises(ValueError):midpoint_state(left,{'other':torch.ones(2)})
        with self.assertRaises(ValueError):midpoint_state({'buffer':torch.tensor(1)},{'buffer':torch.tensor(2)})

    def test_kl_zero_and_detached_teacher(self):
        x=torch.tensor([-100.,0.,100.],requires_grad=True)
        teacher=x.detach().clone().requires_grad_(True)
        loss=bernoulli_kl(x,teacher)
        self.assertAlmostEqual(loss.item(),0.,places=7)
        loss.backward(); self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(bernoulli_kl(torch.zeros(3),teacher).item(),0.)


class SmokeReloadTests(unittest.TestCase):
    def test_fresh_reload_restores_students_and_all_statistics_banks(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from reproductions.three_directions.run import reload_smoke_checkpoint

        def make_fixture(_payload, _history):
            return nn.Sequential(nn.BatchNorm2d(2), nn.Conv2d(2, 1, 1))

        for banks in (0, 1, 4):
            with self.subTest(banks=banks), tempfile.TemporaryDirectory() as directory:
                model = make_fixture(None, 1)
                if banks:
                    model = StatisticsModel(model, banks)
                    for layer in model.statistic_layers():
                        layer.running_mean.fill_(3)
                        layer.running_var.fill_(5)
                        layer.count.fill_(7)
                path = Path(directory) / 'checkpoint.pt'
                state = {key: value.clone() for key, value in model.state_dict().items()}
                saved = dict(history=1, bn_banks=banks, smoke=True, state_dict=state)
                torch.save(saved, path)
                with patch('reproductions.three_directions.run.make_model', side_effect=make_fixture):
                    reloaded = reload_smoke_checkpoint(model, {}, 1, banks, path)
                    self.assertIsNot(reloaded, model)
                    for key, value in model.state_dict().items():
                        self.assertTrue(torch.equal(reloaded.state_dict()[key], value), key)
                    first = next(iter(state))
                    state[first].add_(1)
                    torch.save(saved, path)
                    with self.assertRaisesRegex(ValueError, 'smoke reload tensor differs'):
                        reload_smoke_checkpoint(model, {}, 1, banks, path)


if __name__=='__main__':unittest.main()
