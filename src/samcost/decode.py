"""Ejecución del prompt encoder + mask decoder de SAM sobre embeddings en caché."""
import json

import numpy as np
import torch
import torch.nn.functional as F

from .common import CACHE


def predict_masks_batched(dec, image_embeddings, image_pe, sparse, dense):
    """Igual que MaskDecoder.predict_masks pero con un embedding de imagen DISTINTO por elemento del lote
    (el original replica una sola imagen para varios prompts con repeat_interleave). Devuelve las 4 máscaras
    (token 0 = salida única; tokens 1-3 = salida multimáscara) y las 4 predicciones de IoU."""
    b = sparse.size(0)
    output_tokens = torch.cat([dec.iou_token.weight, dec.mask_tokens.weight], dim=0)
    tokens = torch.cat((output_tokens.unsqueeze(0).expand(b, -1, -1), sparse), dim=1)
    src = image_embeddings + dense
    pos_src = image_pe.expand(b, -1, -1, -1)
    _, c, h, w = src.shape
    hs, src = dec.transformer(src, pos_src, tokens)
    iou_token_out = hs[:, 0, :]
    mask_tokens_out = hs[:, 1:(1 + dec.num_mask_tokens), :]
    src = src.transpose(1, 2).reshape(b, c, h, w)
    up = dec.output_upscaling(src)
    hyper_in = torch.stack([dec.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                            for i in range(dec.num_mask_tokens)], dim=1)
    _, c2, h2, w2 = up.shape
    masks = (hyper_in @ up.reshape(b, c2, h2 * w2)).reshape(b, -1, h2, w2)
    return masks, dec.iou_prediction_head(iou_token_out)


class CachedSAM:
    def __init__(self, sam):
        from segment_anything.utils.transforms import ResizeLongestSide

        self.sam = sam
        self.tf = ResizeLongestSide(1024)
        with torch.no_grad():
            self.pe = sam.prompt_encoder.get_dense_pe()
        self.meta = json.load(open(CACHE / "meta.json"))

    def emb(self, iid):
        e = torch.from_numpy(np.load(CACHE / "emb" / f"{iid}.npy").astype(np.float32))[None]
        m = self.meta[iid]
        return e, tuple(m["original_size"]), tuple(m["input_size"])

    def require_ids(self, ids):
        """Fail rather than silently evaluating a subset of a declared split."""
        missing = [i for i in ids if i not in self.meta or not (CACHE / "emb" / f"{i}.npy").is_file()]
        if missing:
            raise FileNotFoundError(f"Faltan {len(missing)} embeddings/metadatos: {missing[:5]}. Ejecute 02_encode.py.")
        return ids

    def no_mask_dense(self, b=1):
        return self.sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(b, -1, 64, 64)

    @torch.no_grad()
    def decode(self, emb, osz, points=None, labels=None, box=None, mask_input=None, multimask=True):
        """points: (N,2) en coordenadas de la imagen original (x,y); labels (N,) 1=positivo 0=negativo;
        box: (4,) xyxy original. Un solo conjunto de prompts -> una predicción (B=1).
        Devuelve logits de baja resolución (1,C,256,256) y IoU predicho (1,C)."""
        pts = None
        if points is not None and len(points):
            c = self.tf.apply_coords(np.asarray(points, float), osz)
            pts = (torch.as_tensor(c, dtype=torch.float)[None], torch.as_tensor(labels, dtype=torch.int)[None])
        bx = None
        if box is not None:
            bx = torch.as_tensor(self.tf.apply_boxes(np.asarray(box, float)[None], osz), dtype=torch.float)
        sp, de = self.sam.prompt_encoder(points=pts, boxes=bx, masks=mask_input)
        return self.sam.mask_decoder(image_embeddings=emb, image_pe=self.pe, sparse_prompt_embeddings=sp,
                                     dense_prompt_embeddings=de, multimask_output=multimask)

    @torch.no_grad()
    def decode_all4(self, emb, sparse, dense=None):
        """Las 4 salidas del decoder para embeddings de prompt dados (sparse: (1,k,256))."""
        dense = self.no_mask_dense(1) if dense is None else dense
        return predict_masks_batched(self.sam.mask_decoder, emb, self.pe, sparse, dense)

    def full(self, lowres, osz, isz):
        """Logits a resolución original (como SamPredictor)."""
        return self.sam.postprocess_masks(lowres, isz, osz)
