import importlib.util
from pathlib import Path
import unittest
import numpy as np

P=Path(__file__).resolve().parents[1]/'scripts/32_negative_specificity.py'
S=importlib.util.spec_from_file_location('negative_specificity',P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)

class NegativeSpecificityTests(unittest.TestCase):
    def test_component_metrics(self):
        x=np.zeros((10,10),bool);x[0:2,0:2]=1;x[5:8,5:8]=1
        area,largest=M.component_metrics(x)
        self.assertAlmostEqual(area,.13);self.assertAlmostEqual(largest,.09)
        self.assertEqual(M.component_metrics(np.zeros((3,4),bool)),(0.0,0.0))
    def test_bootstrap_constant(self):
        r=M.bootstrap_mean([.25]*5,n=100)
        self.assertEqual(r['mean'],.25);self.assertEqual(r['ci95'],[.25,.25])
    def test_method_averages_seeds_per_image(self):
        rows=[]
        for value in [0.0,.02]:
            row={}
            for s in range(3):row[f'x_area_s{s}']=value;row[f'x_largest_s{s}']=value
            rows.append(row)
        out,av,_,_=M.summarize_method(rows,'x')
        self.assertTrue(np.allclose(av,[0,.02]));self.assertAlmostEqual(out['area_fraction']['mean'],.01)
        self.assertAlmostEqual(out['area_threshold_rates']['0.01']['mean'],.5)

if __name__=='__main__':unittest.main()
