"""Synthetic checks; these do not substitute for inference with the official checkpoint."""
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import shutil
import uuid
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from samcost.learned import ConditionalPrompt, seg_loss
from samcost.decode import predict_masks_batched


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PromptEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.no_mask_embed = nn.Embedding(1, 256)
        self.register_buffer("position", torch.randn(1, 256, 64, 64))

    def get_dense_pe(self):
        return self.position


class FrozenSAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_encoder = PromptEncoder()
        self.mask_decoder = nn.Linear(256, 1)
        for p in self.parameters():
            p.requires_grad_(False)


class PhaseBTests(unittest.TestCase):
    def test_metric_alignment_rejects_incomplete_and_duplicate_images(self):
        module = script("13_summarize_phase_b")
        path = Mock(spec=Path)
        path.open.side_effect = lambda **kwargs: io.StringIO("id,dice\nb,0.7\na,0.2\n")
        np.testing.assert_array_equal(module.read_values(path, "dice", ["a", "b"]), [0.2, 0.7])
        path.open.side_effect = lambda **kwargs: io.StringIO("id,dice\na,0.2\na,0.7\n")
        with self.assertRaises(ValueError):
            module.read_values(path, "dice", ["a", "b"])
        path.open.side_effect = lambda **kwargs: io.StringIO("id,dice\na,0.2\n")
        with self.assertRaises(ValueError):
            module.read_values(path, "dice", ["a", "b"])

    @unittest.skipUnless(importlib.util.find_spec("segment_anything"), "SAM oficial no instalado")
    def test_batched_decoder_matches_official_independent_images(self):
        from segment_anything.modeling import MaskDecoder, TwoWayTransformer
        torch.manual_seed(3)
        torch.set_num_threads(2)
        dec = MaskDecoder(transformer_dim=256,
                          transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8))
        dec.eval()
        emb = torch.randn(2, 256, 64, 64)
        pe = torch.randn(1, 256, 64, 64)
        dense = torch.randn(2, 256, 64, 64)
        for k in (0, 4):
            sparse = torch.randn(2, k, 256)
            with torch.no_grad():
                actual, quality = predict_masks_batched(dec, emb, pe, sparse, dense)
                for i in range(2):
                    masks, scores = dec.predict_masks(emb[i:i+1], pe, sparse[i:i+1], dense[i:i+1])
                    torch.testing.assert_close(actual[i:i+1], masks, atol=2e-5, rtol=1e-4)
                    torch.testing.assert_close(quality[i:i+1], scores, atol=2e-5, rtol=1e-4)

    @unittest.skipUnless(importlib.util.find_spec("segment_anything"), "SAM oficial no instalado")
    def test_conditional_gradients_through_official_frozen_decoder(self):
        from segment_anything.modeling import MaskDecoder, TwoWayTransformer
        torch.manual_seed(4)
        torch.set_num_threads(2)
        sam = FrozenSAM()
        sam.mask_decoder = MaskDecoder(transformer_dim=256,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8))
        for p in sam.parameters():
            p.requires_grad_(False)
        model = ConditionalPrompt(sam, init_tokens=torch.randn(4, 256))
        logits, _ = model(torch.randn(1, 256, 64, 64))
        target = torch.zeros_like(logits)
        target[:, 64:192, 64:192] = 1
        seg_loss(logits, target, torch.ones_like(logits)).backward()
        self.assertGreater(model.attention.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in sam.parameters()))

    def test_conditional_prompts_depend_on_image_and_train_with_frozen_sam(self):
        torch.manual_seed(2)
        torch.set_num_threads(2)
        sam = FrozenSAM()
        model = ConditionalPrompt(sam, init_tokens=torch.randn(4, 256))
        before = {n: p.detach().clone() for n, p in sam.named_parameters()}
        model.train()
        self.assertFalse(sam.training)
        emb = torch.randn(2, 256, 64, 64)
        sparse = model.prompt_embeddings(emb)
        self.assertEqual(tuple(sparse.shape), (2, 4, 256))
        self.assertGreater((sparse[0] - sparse[1]).abs().max().item(), 1e-6)

        def decoder(dec, features, pe, prompts, dense):
            # A differentiable stand-in isolates the prompt-to-output training path.
            value = dec(prompts).mean(1)
            return value[:, :, None, None].expand(-1, 4, 8, 8), value.expand(-1, 4)

        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        with patch("samcost.learned.predict_masks_batched", side_effect=decoder):
            logits, _ = model(emb)
            loss = seg_loss(logits, torch.ones_like(logits), torch.ones_like(logits))
            loss.backward()
        self.assertGreater(model.attention.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.tokens.grad.abs().sum().item(), 0)
        opt.step()
        for n, p in sam.named_parameters():
            self.assertIsNone(p.grad)
            self.assertTrue(torch.equal(p, before[n]))

    def test_padding_does_not_change_loss(self):
        target = torch.zeros(1, 8, 8)
        target[:, :4, :4] = 1
        valid = torch.zeros_like(target)
        valid[:, :4, :] = 1
        a = torch.zeros_like(target)
        b = a.clone()
        b[:, 4:, :] = 100
        self.assertTrue(torch.allclose(seg_loss(a, target, valid), seg_loss(b, target, valid)))

    def test_uniform_attention_has_unit_lift_with_padding(self):
        module = script("05b_attention_stats")
        valid = np.zeros((64, 64), np.float32)
        valid[:32] = 1
        lab = np.zeros_like(valid)
        lab[:16] = 1
        result = module.stats_of(np.ones_like(valid), lab, valid)
        self.assertAlmostEqual(result["lift"], 1.0)
        self.assertAlmostEqual(result["mass_in_padding"], 0.5)

    def test_cache_rejects_precision_change_and_missing_blocks(self):
        module = script("02_encode")
        emb, blocks = Mock(spec=Path), Mock(spec=Path)
        emb.is_file.return_value = True
        blocks.is_file.return_value = False
        meta = {"precision": "fp32", "storage": "fp16"}
        self.assertTrue(module.cache_matches(meta, meta, emb, blocks, False))
        self.assertFalse(module.cache_matches(meta, {**meta, "storage": "fp32"}, emb, blocks, False))
        self.assertFalse(module.cache_matches(meta, meta, emb, blocks, True))

    def test_run_dir_redirects_results_and_figures(self):
        code = "import samcost.common as c; print(c.RESULTS); print(c.FIGURES)"

        def paths(run_dir):
            env = {k: v for k, v in os.environ.items() if k != "SAMCOST_RUN_DIR"}
            if run_dir:
                env["SAMCOST_RUN_DIR"] = str(run_dir)
            out = subprocess.run([sys.executable, "-c", code], cwd=ROOT / "src", env=env,
                                 capture_output=True, text=True, check=True).stdout.splitlines()
            return Path(out[0]), Path(out[1])

        self.assertEqual(paths(None), (ROOT / "results", ROOT / "figures"))
        # Keep subprocess writes within the workspace on restricted Windows setups.
        temp_root = ROOT / "runs"
        temp_root.mkdir(exist_ok=True)
        # Python 3.13 mkdtemp(mode=0o700) can create an ACL inaccessible to
        # restricted subprocesses on Windows. An ordinary workspace directory
        # tests the same routing behavior without relying on that private ACL.
        tmp = temp_root / ("test_routes_" + uuid.uuid4().hex)
        tmp.mkdir()
        try:
            run = tmp / "run_1"
            self.assertEqual(paths(run), (run, run / "figures"))
            self.assertTrue(run.is_dir())
            relative = tmp.relative_to(ROOT) / "relative_run"
            expected = ROOT / relative
            self.assertEqual(paths(relative), (expected, expected / "figures"))
        finally:
            if not tmp.resolve().is_relative_to(temp_root.resolve()):
                raise ValueError("Temporary test path escaped workspace")
            shutil.rmtree(tmp)

    def test_perturbations_clip_coordinates_and_preserve_identity(self):
        module = script("10_prompt_perturbations")
        point = np.array([5., 6.])
        np.testing.assert_array_equal(module.shifted_point(point, 20, 0, (1, 0), (10, 10)), point)
        np.testing.assert_array_equal(module.shifted_point(point, 20, 4, (1, 0), (10, 10)), [9, 6])
        box = np.array([2., 3., 6., 7.])
        np.testing.assert_array_equal(module.expanded_box(box, 1, (10, 10)), box)
        np.testing.assert_array_equal(module.expanded_box(box, 10, (10, 10)), [0, 0, 9, 9])


if __name__ == "__main__":
    unittest.main()
