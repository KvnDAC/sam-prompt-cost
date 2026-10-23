"""Checks for new architecture controls and interruption-sensitive training state."""
import importlib.util
from pathlib import Path
import sys
import unittest
import copy
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from samcost.light_baselines import build_model
from samcost.learned import seg_loss
spec=importlib.util.spec_from_file_location('comparison',ROOT/'scripts/20_compare_light_models.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)


class LightComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_new_models_learn_visible_foreground_and_odd_shapes(self):
        for name in ['fcn_lite','ds_unet']:
            with self.subTest(model=name):
                torch.manual_seed(11)
                model=build_model(name,4)
                target=torch.zeros(1,33,49);target[:,8:25,12:37]=1
                image=target[:,None].repeat(1,3,1,1)
                opt=torch.optim.Adam(model.parameters(),lr=.01)
                for _ in range(90):
                    opt.zero_grad(set_to_none=True)
                    logits=model(image)
                    self.assertEqual(logits.shape,target.shape)
                    loss=seg_loss(logits,target,torch.ones_like(target));loss.backward();opt.step()
                pred=(model(image)>0).float()
                score=2*(pred*target).sum()/(pred.sum()+target.sum())
                self.assertGreater(float(score),.95)

    def test_restore_optimizer_rng_and_sample_order(self):
        torch.manual_seed(3)
        model=build_model('fcn_lite',4);opt=torch.optim.Adam(model.parameters(),lr=.001)
        generator=torch.Generator().manual_seed(7)
        def step():
            permutation=torch.randperm(4,generator=generator)
            x=torch.randn(2,3,32,32)
            opt.zero_grad(set_to_none=True)
            loss=model(x).square().mean();loss.backward();opt.step()
            return permutation,loss.detach()
        step();saved=copy.deepcopy(comparison.checkpoint_state(model,opt,generator))
        order,loss=step();expected=copy.deepcopy(model.state_dict())
        comparison.restore_state(saved,model,opt,generator)
        order2,loss2=step()
        self.assertTrue(torch.equal(order,order2));self.assertTrue(torch.equal(loss,loss2))
        for key,value in expected.items():self.assertTrue(torch.equal(value,model.state_dict()[key]),key)

    def test_pairing_uses_ids_and_rejects_mismatches(self):
        a=[{'id':'a','dice':.8},{'id':'b','dice':.4}]
        b=[{'id':'b','dice':.2},{'id':'a','dice':.6}]
        self.assertAlmostEqual(comparison.paired_difference(a,b)[0],.2)
        with self.assertRaises(ValueError):comparison.paired_difference(a,b[:1])
        with self.assertRaises(ValueError):comparison.paired_difference(a,[b[0],b[0]])

    def test_disjoint_splits_required(self):
        train=[f't{i}' for i in range(400)];dev=[f'd{i}' for i in range(100)]
        comparison.validate_ids(train,dev)
        dev[0]=train[0]
        with self.assertRaises(ValueError):comparison.validate_ids(train,dev)


if __name__=='__main__':unittest.main()
