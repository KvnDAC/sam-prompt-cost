"""Captura de los mapas de atención token→imagen del mask decoder (para ver hacia dónde «mira» cada token)."""
import math
from contextlib import contextmanager

import torch


@contextmanager
def capture_token_to_image(sam):
    """Durante el bloque, guarda en `store[name]` la atención (B, heads, n_tokens, 4096) de las capas
    token→imagen del decoder: layer0, layer1 y final."""
    t = sam.mask_decoder.transformer
    mods = {"layer0": t.layers[0].cross_attn_token_to_image, "layer1": t.layers[1].cross_attn_token_to_image,
            "final": t.final_attn_token_to_image}
    store, originals = {}, {}

    def make(name, m):
        def fwd(q, k, v):
            q2, k2, v2 = m.q_proj(q), m.k_proj(k), m.v_proj(v)
            q2, k2, v2 = (m._separate_heads(z, m.num_heads) for z in (q2, k2, v2))
            attn = torch.softmax(q2 @ k2.permute(0, 1, 3, 2) / math.sqrt(q2.shape[-1]), dim=-1)
            store[name] = attn.detach()
            return m.out_proj(m._recombine_heads(attn @ v2))
        return fwd

    for name, m in mods.items():
        originals[name] = m.forward
        m.forward = make(name, m)
    try:
        yield store
    finally:
        for name, m in mods.items():
            m.forward = originals[name]
