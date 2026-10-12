"""Intervención en el núcleo: tokens de prompt aprendidos con SAM completamente congelado.

El prompt encoder se reemplaza por k vectores aprendibles (k×256) que entran al mask decoder exactamente
donde entrarían los embeddings de puntos/cajas. No dependen de la imagen: pueden codificar QUÉ segmentar,
pero no DÓNDE está el objeto en cada imagen.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decode import predict_masks_batched


class LearnedPrompt(nn.Module):
    def __init__(self, sam, k=4, init_std=None, learn_dense=False, out_token=0, init_tokens=None):
        super().__init__()
        self.sam = sam
        if init_std is None and init_tokens is None:  # escala de un prompt real
            with torch.no_grad():
                sp, _ = sam.prompt_encoder(points=(torch.tensor([[[512.0, 512.0]]]), torch.ones(1, 1, dtype=torch.int)),
                                           boxes=None, masks=None)
            init_std = float(sp.std())
        if init_tokens is not None:  # inicialización basada en datos (p. ej., clic medio en el centro de los pólipos)
            self.tokens = nn.Parameter(init_tokens.reshape(1, k, 256).clone().float())
        else:
            self.tokens = nn.Parameter(torch.randn(1, k, 256) * init_std)
        nm = sam.prompt_encoder.no_mask_embed.weight.detach().clone().reshape(1, 256, 1, 1)
        self.dense = nn.Parameter(nm) if learn_dense else None
        self.register_buffer("no_mask", nm)
        self.register_buffer("pe", sam.prompt_encoder.get_dense_pe().detach())
        self.out_token = out_token

    def forward(self, emb):
        b = emb.shape[0]
        sparse = self.tokens.expand(b, -1, -1)
        dense = (self.dense if self.dense is not None else self.no_mask).expand(b, -1, 64, 64)
        masks, iou = predict_masks_batched(self.sam.mask_decoder, emb, self.pe, sparse, dense)
        return masks[:, self.out_token], iou[:, self.out_token]

    def train(self, mode=True):
        super().train(mode)
        self.sam.eval()
        return self


class ConditionalPrompt(LearnedPrompt):
    """Residual image-conditioned prompts; SAM frozen, pooled spatial keys with SAM PE."""

    def __init__(self, sam, k=4, init_tokens=None):
        super().__init__(sam, k=k, init_tokens=init_tokens)
        self.feature_norm = nn.LayerNorm(256)
        self.query_norm = nn.LayerNorm(256)
        self.attention = nn.MultiheadAttention(256, 8, batch_first=True)
        self.residual = nn.Linear(256, 256)
        # Start close to static prompts without blocking gradients to attention.
        nn.init.normal_(self.residual.weight, std=1e-3)
        nn.init.zeros_(self.residual.bias)

    def prompt_embeddings(self, emb):
        b = emb.shape[0]
        features = F.adaptive_avg_pool2d(emb, (16, 16)).flatten(2).transpose(1, 2)
        position = F.adaptive_avg_pool2d(self.pe, (16, 16)).flatten(2).transpose(1, 2)
        query = self.tokens.expand(b, -1, -1)
        values = self.feature_norm(features)
        update, _ = self.attention(self.query_norm(query), values + position, values, need_weights=False)
        return query + self.residual(update)

    def forward(self, emb):
        b = emb.shape[0]
        sparse = self.prompt_embeddings(emb)
        dense = self.no_mask.expand(b, -1, 64, 64)
        masks, quality = predict_masks_batched(self.sam.mask_decoder, emb, self.pe, sparse, dense)
        return masks[:, self.out_token], quality[:, self.out_token]


def seg_loss(logits, target, valid):
    """BCE + Dice suave sobre la región válida (sin padding) del marco de 256×256."""
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum((1, 2)) / valid.sum((1, 2))
    p = torch.sigmoid(logits) * valid
    inter = (p * target).sum((1, 2))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2)) + (target * valid).sum((1, 2)) + 1)
    return (bce + dice).mean()


def valid_mask(input_size, size=256):
    h, w = input_size
    v = np.zeros((size, size), np.float32)
    v[: int(round(h * size / 1024)), : int(round(w * size / 1024))] = 1
    return v
