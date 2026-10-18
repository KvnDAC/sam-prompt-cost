"""The spatial prior must be derived only from the supplied training masks."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prior',ROOT/'scripts/22_fixed_spatial_prior.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)


class FixedPriorTests(unittest.TestCase):
    def test_normalized_box_and_point_use_only_supplied_ids(self):
        mask=np.zeros((11,21),bool);mask[2:9,4:17]=True
        calls=[]
        def load(i):
            calls.append(i);self.assertIn(i,['train_a','train_b']);return mask
        with patch.object(prior,'load_mask',side_effect=load):
            box,point=prior.derive_prior(['train_a','train_b'])
        np.testing.assert_allclose(box,[.2,.2,.8,.8])
        self.assertTrue(np.all(point>=box[:2]) and np.all(point<=box[2:]))
        self.assertEqual(calls,['train_a','train_b'])

    def test_empty_training_mask_is_rejected(self):
        with patch.object(prior,'load_mask',return_value=np.zeros((16,16),bool)):
            with self.assertRaises(ValueError):prior.derive_prior(['train'])


if __name__=='__main__':unittest.main()
