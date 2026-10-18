"""Geometry and trainability checks for the independent RGB segmentation baseline."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from samcost.unet import UNet, prepare_arrays, restore_logits
from samcost.learned import seg_loss


class UNetTests(unittest.TestCase):
    def test_can_learn_a_visible_foreground_region(self):
        torch.set_num_threads(1)
        torch.manual_seed(2)
        model = UNet(base_channels=4)
        target = torch.zeros(1, 32, 48)
        target[:, 8:24, 12:36] = 1
        image = target[:, None].repeat(1, 3, 1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            seg_loss(logits, target, torch.ones_like(target)).backward()
            optimizer.step()
        with torch.no_grad():
            pred = model(image) > 0
        score = 2 * (pred * target).sum() / (pred.sum() + target.sum())
        self.assertGreater(score.item(), 0.95)

    def test_rectangular_mask_roundtrip_excludes_padding(self):
        image = np.zeros((80, 160, 3), dtype=np.uint8)
        mask = np.zeros((80, 160), dtype=bool)
        mask[20:60, 40:120] = True
        image[mask] = 255
        x, target, valid, size = prepare_arrays(image, mask, 128)
        self.assertEqual(size, (64, 128))
        self.assertEqual(valid.sum().item(), 64 * 128)
        self.assertEqual(x[:, 64:].abs().sum().item(), 0)
        logits = target * 20 - 10
        logits[64:] = 100  # Would corrupt output if padding were resized with the image.
        restored = (restore_logits(logits, size, mask.shape) > 0).numpy()
        np.testing.assert_array_equal(restored, mask)

    def test_logits_shape_and_gradients_reach_both_ends(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        model = UNet(base_channels=4)
        x = torch.randn(2, 3, 33, 49)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 33, 49))
        target = torch.zeros_like(logits)
        target[:, 8:24, 12:36] = 1
        seg_loss(logits, target, torch.ones_like(target)).backward()
        self.assertGreater(model.encoders[0][0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.output.weight.grad.abs().sum().item(), 0)

    def test_group_norm_variant_has_finite_logits_and_encoder_gradients(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        model = UNet(base_channels=4, normalization="group")
        logits = model(torch.randn(2, 3, 32, 48))
        self.assertTrue(torch.isfinite(logits).all())
        seg_loss(logits, torch.ones_like(logits), torch.ones_like(logits)).backward()
        self.assertGreater(model.encoders[0][0].weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
