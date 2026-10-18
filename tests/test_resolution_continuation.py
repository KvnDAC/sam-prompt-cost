"""Resolution interventions must preserve the common parent, not its old score."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import shutil
import unittest
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('resolution26', ROOT/'scripts/26_unet_resolution_continuation.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class ResolutionTests(unittest.TestCase):
    def test_matched_parent_rng_adam_and_resolution_specific_selection(self):
        torch.set_num_threads(1)
        folder = ROOT/'runs'/f'test_resolution_{uuid.uuid4().hex}'
        folder.mkdir(parents=True)
        try:
            torch.manual_seed(7)
            model = study.light.build_model('unet', 2)
            opt = torch.optim.Adam(model.parameters(), lr=.001)
            opt.zero_grad(); model(torch.randn(2,3,32,32)).square().mean().backward(); opt.step()
            parent = study.light.checkpoint_state(model, opt, torch.Generator().manual_seed(17))
            parent.update(epoch=30, best=.999, best_epoch=28)
            path = folder/'parent.pt'; torch.save(parent, path)
            masks = [np.ones((24,32), bool)]
            states = []
            for size in [32,64]:
                args = SimpleNamespace(width=2, size=size, batch_size=1)
                sample = study.light.prepare_arrays(np.full((24,32,3),120,np.uint8), masks[0], size)
                data = SimpleNamespace(x=sample[0][None], sizes=[sample[3]])
                out = folder/f'size{size}'
                study.initialize_stage(path, out, args, data, masks, ['image'])
                state = torch.load(out/'resume.pt', weights_only=False)
                states.append(state)
                expected = study.light.evaluate(model, ['image'], data, masks, 1)[0]['dice']
                self.assertEqual(state['best'], expected)
                self.assertEqual(state['epoch'],0); self.assertEqual(state['history'],[])
                self.assertNotEqual(state['best'],parent['best'])
                # Resume is idempotent, including saved RNG and selection.
                digest = study.sha(out/'resume.pt')
                study.initialize_stage(path, out, args, data, masks, ['image'])
                self.assertEqual(digest,study.sha(out/'resume.pt'))
                for key,value in parent['model'].items():
                    self.assertTrue(torch.equal(value,state['model'][key]))
                self.assertTrue(torch.equal(parent['torch_rng'],state['torch_rng']))
                self.assertTrue(torch.equal(parent['loader_rng'],state['loader_rng']))
                for key, values in parent['optimizer']['state'].items():
                    for name,value in values.items():
                        self.assertTrue(torch.equal(value,state['optimizer']['state'][key][name]))
            self.assertTrue(torch.equal(states[0]['torch_rng'],states[1]['torch_rng']))
            parent['epoch']=29; torch.save(parent,path)
            with self.assertRaises(ValueError):
                study.initialize_stage(path,folder/'wrong',args,data,masks,['image'])
        finally:
            self.assertTrue(folder.resolve().is_relative_to((ROOT/'runs').resolve()))
            shutil.rmtree(folder)


if __name__ == '__main__':
    unittest.main()
