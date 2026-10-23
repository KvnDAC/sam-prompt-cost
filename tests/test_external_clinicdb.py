import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('external30',ROOT/'scripts/30_external_clinicdb.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)

class ExternalTests(unittest.TestCase):
    def test_sequence_boundaries(self):
        self.assertEqual(study.sequence(1),1);self.assertEqual(study.sequence(25),1)
        self.assertEqual(study.sequence(26),2);self.assertEqual(study.sequence(612),29)
        with self.assertRaises(ValueError):study.sequence(0)
    def test_cluster_interval_constant_and_macro(self):
        sequences=np.repeat(np.arange(1,30),2);values=np.ones(58)
        got=study.cluster_interval(values,sequences,n=100)
        self.assertEqual(got['frame_mean'],1);self.assertEqual(got['sequence_macro_mean'],1)
        self.assertEqual(got['cluster_ci95'],[1,1])
    def test_all_declared_frames_map_to_29_nonempty_sequences(self):
        groups=[study.sequence(i) for i in range(1,613)]
        self.assertEqual(set(groups),set(range(1,30)))
        self.assertEqual([groups.count(i) for i in range(1,30)],np.diff([0,*study.BOUNDS]).tolist())

if __name__=='__main__':unittest.main()
