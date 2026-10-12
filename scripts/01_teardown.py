"""Fase A — Disección instrumentada de SAM ViT-B: parámetros y FLOPs por bloque, latencia y memoria.

Uso:  python scripts/01_teardown.py            (todo)
      python scripts/01_teardown.py --mem_probe enc_fp32   (uso interno: mide memoria en un proceso limpio)
Salida: results/teardown.json, results/teardown_blocks.csv
"""
import argparse, csv, json, resource, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from samcost.common import RESULTS, env_info, load_sam, save_json, set_threads

GLOBAL = (2, 5, 8, 11)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def param_table(sam):
    enc, pe, dec = sam.image_encoder, sam.prompt_encoder, sam.mask_decoder
    rows = [("encoder", "patch_embed (conv 16x16, 3→768)", nparams(enc.patch_embed)),
            ("encoder", "pos_embed absoluto (64×64×768)", enc.pos_embed.numel())]
    for i, b in enumerate(enc.blocks):
        kind = "global" if i in GLOBAL else "ventana 14×14"
        rows.append(("encoder", f"block {i:02d} ({kind})", nparams(b)))
    rows.append(("encoder", "neck (1×1 768→256, 3×3 256→256, 2×LN)", nparams(enc.neck)))
    rows += [("prompt_encoder", "point_embeddings (4×256) + not_a_point", sum(p.numel() for p in pe.point_embeddings.parameters()) + pe.not_a_point_embed.weight.numel()),
             ("prompt_encoder", "mask_downscaling (convs)", nparams(pe.mask_downscaling)),
             ("prompt_encoder", "no_mask_embed (1×256)", pe.no_mask_embed.weight.numel()),
             ("prompt_encoder", "pe_layer (Fourier aleatorio, buffer no entrenable)", 0)]
    t = dec.transformer
    for i, l in enumerate(t.layers):
        rows.append(("mask_decoder", f"two-way layer {i} (self-attn tokens, t→i, MLP, i→t)", nparams(l)))
    rows += [("mask_decoder", "final attn token→imagen + LN", nparams(t.final_attn_token_to_image) + nparams(t.norm_final_attn)),
             ("mask_decoder", "output_upscaling (2 ConvTranspose ×2)", nparams(dec.output_upscaling)),
             ("mask_decoder", "hypernetwork MLPs (4 × MLP 256→32)", nparams(dec.output_hypernetworks_mlps)),
             ("mask_decoder", "iou_prediction_head (MLP 256→4)", nparams(dec.iou_prediction_head)),
             ("mask_decoder", "iou_token + 4 mask_tokens", dec.iou_token.weight.numel() + dec.mask_tokens.weight.numel())]
    return rows


def flops_by_module(fn):
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        fn()
    counts = fc.get_flop_counts()
    return {k: sum(v.values()) for k, v in counts.items()}, fc.get_total_flops()


def decoder_call(sam, emb, n_points=1, box=False, multimask=True):
    pts = None
    if n_points:
        coords = torch.rand(n_points, 1, 2) * 1024
        labels = torch.ones(n_points, 1, dtype=torch.int)
        pts = (coords, labels)
    boxes = torch.tensor([[100.0, 100.0, 600.0, 600.0]]) if box else None
    sp, de = sam.prompt_encoder(points=pts, boxes=boxes, masks=None)
    return sam.mask_decoder(image_embeddings=emb, image_pe=sam.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sp, dense_prompt_embeddings=de, multimask_output=multimask)


def timeit(fn, n=5, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts)), float(np.max(ts))


def mem_probe(kind):
    """Memoria pico de activaciones: RSS muestreado cada 2 ms durante el forward, menos el RSS en reposo
    tras cargar el modelo y devolver al SO la memoria del checkpoint (gc + malloc_trim)."""
    import ctypes, gc, threading
    import psutil
    set_threads(2)
    sam = load_sam()
    x = torch.randn(1, 3, 1024, 1024)
    emb = torch.randn(1, 256, 64, 64)
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    proc = psutil.Process()
    base = proc.memory_info().rss
    peak = [base]
    stop = threading.Event()
    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.002)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    with torch.no_grad():
        if kind == "enc_fp32":
            sam.image_encoder(x)
        elif kind == "enc_bf16":
            with torch.autocast("cpu", dtype=torch.bfloat16):
                sam.image_encoder(x)
        elif kind == "dec_1pt":
            for _ in range(5):
                decoder_call(sam, emb, 1)
        elif kind == "dec_64pt":
            decoder_call(sam, emb, 64)
    stop.set(); th.join()
    print(json.dumps({"kind": kind, "rss_idle_MB": base / 2**20, "rss_peak_MB": peak[0] / 2**20,
                      "activation_peak_MB": (peak[0] - base) / 2**20}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem_probe")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--only_memory", action="store_true")
    a = ap.parse_args()
    if a.mem_probe:
        return mem_probe(a.mem_probe)
    if a.only_memory:
        out = json.load(open(RESULTS / "teardown.json"))
        mem = {}
        for kind in ["enc_fp32", "enc_bf16", "dec_1pt", "dec_64pt"]:
            r = subprocess.run([sys.executable, __file__, "--mem_probe", kind], capture_output=True, text=True)
            mem[kind] = json.loads(r.stdout.strip().splitlines()[-1])
        out["memory"] = mem
        save_json(out, RESULTS / "teardown.json")
        print(json.dumps(mem, indent=1))
        return

    threads = set_threads(2)
    sam = load_sam()
    out = {"env": env_info(), "threads": threads}

    # 1) Parámetros
    rows = param_table(sam)
    tot = nparams(sam)
    out["params"] = {"total": tot, "image_encoder": nparams(sam.image_encoder),
                     "prompt_encoder": nparams(sam.prompt_encoder), "mask_decoder": nparams(sam.mask_decoder)}

    # 2) FLOPs (convención: 1 MAC = 2 FLOPs; solo matmul/conv, como FlopCounterMode)
    x = torch.randn(1, 3, 1024, 1024)
    enc_f, enc_tot = flops_by_module(lambda: sam.image_encoder(x))
    emb = torch.randn(1, 256, 64, 64)
    dec_f, dec_tot = flops_by_module(lambda: decoder_call(sam, emb, 1))
    _, dec_box = flops_by_module(lambda: decoder_call(sam, emb, 0, box=True))
    _, dec_64 = flops_by_module(lambda: decoder_call(sam, emb, 64))
    _, dec_none = flops_by_module(lambda: decoder_call(sam, emb, 0))
    blk = []
    for i in range(12):
        k = f"ImageEncoderViT.blocks.{i}"
        attn = enc_f.get(k + ".attn", 0)
        qkvproj = enc_f.get(k + ".attn.qkv", 0) + enc_f.get(k + ".attn.proj", 0)
        blk.append({"block": i, "type": "global" if i in GLOBAL else "window", "GFLOPs": enc_f[k] / 1e9,
                    "attn_matmul_GFLOPs": (attn - qkvproj) / 1e9, "qkv_proj_GFLOPs": qkvproj / 1e9,
                    "mlp_GFLOPs": enc_f.get(k + ".mlp", 0) / 1e9})
    out["flops"] = {
        "encoder_GFLOPs": enc_tot / 1e9, "patch_embed_GFLOPs": enc_f.get("ImageEncoderViT.patch_embed", 0) / 1e9,
        "neck_GFLOPs": enc_f.get("ImageEncoderViT.neck", 0) / 1e9, "blocks": blk,
        "decoder_1point_GFLOPs": dec_tot / 1e9, "decoder_box_GFLOPs": dec_box / 1e9,
        "decoder_noprompt_GFLOPs": dec_none / 1e9, "decoder_64points_GFLOPs": dec_64 / 1e9,
        "decoder_modules_1point_GFLOPs": {k.replace("MaskDecoder.", ""): v / 1e9 for k, v in dec_f.items()
                                          if k.count(".") <= 2 and k.startswith("MaskDecoder")},
    }
    g = np.mean([b["GFLOPs"] for b in blk if b["type"] == "global"])
    w = np.mean([b["GFLOPs"] for b in blk if b["type"] == "window"])
    out["flops"]["global_over_window_ratio"] = g / w
    out["flops"]["encoder_share_1prompt"] = enc_tot / (enc_tot + dec_tot)

    # 3) Latencia (CPU, 2 hilos)
    with torch.no_grad():
        lat = {}
        lat["encoder_fp32_s"] = timeit(lambda: sam.image_encoder(x), n=a.reps)
        def enc_bf16():
            with torch.autocast("cpu", dtype=torch.bfloat16):
                return sam.image_encoder(x)
        lat["encoder_bf16_s"] = timeit(enc_bf16, n=a.reps)
        lat["decoder_1point_ms"] = [v * 1e3 for v in timeit(lambda: decoder_call(sam, emb, 1), n=30, warmup=3)]
        lat["decoder_box_ms"] = [v * 1e3 for v in timeit(lambda: decoder_call(sam, emb, 0, box=True), n=30, warmup=3)]
        lat["decoder_noprompt_ms"] = [v * 1e3 for v in timeit(lambda: decoder_call(sam, emb, 0), n=30, warmup=3)]
        lat["decoder_64points_ms"] = [v * 1e3 for v in timeit(lambda: decoder_call(sam, emb, 64), n=10, warmup=2)]
        lowres = torch.randn(1, 1, 256, 256)
        lat["postprocess_to_1024_ms"] = [v * 1e3 for v in timeit(lambda: sam.postprocess_masks(lowres, (1024, 683), (1072, 715)), n=30, warmup=3)]
        # fidelidad bf16 vs fp32 del embedding
        e32 = sam.image_encoder(x)
        e16 = enc_bf16().float()
        cos = torch.nn.functional.cosine_similarity(e32.flatten(1), e16.flatten(1)).item()
        lat["bf16_vs_fp32_cosine"] = cos
        lat["bf16_vs_fp32_rel_err"] = ((e32 - e16).norm() / e32.norm()).item()
    out["latency"] = lat

    # 4) Memoria pico (procesos limpios)
    mem = {}
    for kind in ["enc_fp32", "enc_bf16", "dec_1pt", "dec_64pt"]:
        r = subprocess.run([sys.executable, __file__, "--mem_probe", kind], capture_output=True, text=True)
        mem[kind] = json.loads(r.stdout.strip().splitlines()[-1])
    out["memory"] = mem

    save_json(out, RESULTS / "teardown.json")
    with open(RESULTS / "teardown_params.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["componente", "bloque", "parametros", "porcentaje"])
        for c, b, n in rows:
            wr.writerow([c, b, n, f"{100 * n / tot:.3f}"])
    with open(RESULTS / "teardown_blocks.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(blk[0].keys())); wr.writeheader(); wr.writerows(blk)
    print(json.dumps({k: out[k] for k in ["params", "latency", "memory"]}, indent=1))
    print("encoder GFLOPs %.1f | global/window %.2f | decoder 1pt GFLOPs %.2f" % (enc_tot / 1e9, g / w, dec_tot / 1e9))


if __name__ == "__main__":
    main()
