"""Unit checks for nested label budgets and exact fixed-step batching."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('labels28',ROOT/'scripts/28_label_efficiency.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


class LabelEfficiencyTests(unittest.TestCase):
    def test_nested_subsets_are_reproducible_prefixes(self):
        ids=[f'i{x}' for x in range(400)]
        a=study.nested_subsets(ids);b=study.nested_subsets(ids)
        self.assertEqual(a,b)
        for seed in study.SEEDS:
            self.assertEqual(a[str(seed)]['25'],a[str(seed)]['100'][:25])
            self.assertEqual(a[str(seed)]['100'],a[str(seed)]['400'][:100])
            self.assertEqual(len(set(a[str(seed)]['400'])),400)

    def test_batch_sampler_is_reproducible_and_keeps_padding(self):
        data=type('D',(),{})();data.x=torch.arange(4*3*16*16).reshape(4,3,16,16).float()
        data.y=torch.zeros(4,16,16);data.valid=torch.zeros(4,16,16);data.sizes=[(8,12)]*4
        data.valid[:,:8,:12]=1
        one=study.sample_batch(data,8,torch.Generator().manual_seed(9))
        two=study.sample_batch(data,8,torch.Generator().manual_seed(9))
        for left,right in zip(one,two):self.assertTrue(torch.equal(left,right))
        self.assertTrue(torch.equal(one[2][:,8:,:],torch.zeros_like(one[2][:,8:,:])))

    def test_interval_and_token_shape(self):
        got=study.interval(np.ones(20))
        self.assertEqual(got['mean'],1.);self.assertEqual(got['ci95'],[1.,1.])
        labels,h,w=study.token_labels(np.ones((512,1024),bool),(512,1024))
        self.assertEqual((h,w), (32,64));self.assertEqual(labels.shape,(32,64))


if __name__=='__main__':unittest.main()
