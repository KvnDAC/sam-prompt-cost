"""Prepara Kvasir-SEG con los nombres oficiales y las particiones del proyecto.

Dos fuentes posibles:
  --official_zip  ruta al Kvasir-SEG.zip oficial (https://datasets.simula.no/kvasir-seg/)  [RECOMENDADO]
  --mirror_dir    carpeta de un espejo con archivos renombrados (colonoscopy_XXXX.jpg / mask_XXXX.jpg)
                  + renamed_bboxes.json. Los nombres se restauran haciendo coincidir (alto, ancho, bbox)
                  con kavsir_bboxes.json oficial; el script falla si la correspondencia no es 1 a 1.

Salida: data/Kvasir-SEG/{images,masks}/<id_oficial>.jpg y data/splits/{train,dev,test}.txt
  test  = val.txt oficial de MediaEval 2020 (120 imagenes, publicado por los autores del dataset)
  train/dev = train.txt oficial (880) partido 780/100 con semilla fija (dev = seleccion de hiperparametros)
"""
import argparse, json, os, random, shutil, zipfile, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def key_of(entry):
    b = entry["bbox"]
    return (entry["height"], entry["width"],
            tuple(sorted((x["xmin"], x["ymin"], x["xmax"], x["ymax"]) for x in b)))


def from_mirror(mirror_dir: Path, out: Path, official: dict):
    """El espejo renombro las IMAGENES en orden alfabetico de los ids oficiales, pero su
    renamed_bboxes.json sigue el orden de insercion del json oficial (inconsistente con sus
    propias imagenes: solo 65/1000 tamanos coinciden). Por eso NO se usa ese json: se asigna
    por orden alfabetico y se verifica contra kavsir_bboxes.json oficial (tamano exacto y
    caja de la mascara vs caja oficial)."""
    import numpy as np
    from PIL import Image
    official_sorted = sorted(official)
    files = sorted(p.name for p in (mirror_dir / "images").glob("colonoscopy_*.jpg"))
    assert len(files) == 1000
    (out / "images").mkdir(parents=True, exist_ok=True); (out / "masks").mkdir(parents=True, exist_ok=True)
    mapping, size_ok, box_iou = {}, 0, []
    for f, oid in zip(files, official_sorted):
        n = f.split("_")[-1].split(".")[0]
        im = Image.open(mirror_dir / "images" / f)
        m = np.array(Image.open(mirror_dir / "masks" / f"mask_{n}.jpg").convert("L")) > 127
        e = official[oid]
        size_ok += int(im.size == (e["width"], e["height"]) and m.shape == (e["height"], e["width"]))
        ys, xs = np.where(m)
        mb = (xs.min(), ys.min(), xs.max(), ys.max())
        B = e["bbox"]
        ob = (min(b["xmin"] for b in B), min(b["ymin"] for b in B), max(b["xmax"] for b in B), max(b["ymax"] for b in B))
        iw = max(0, min(mb[2], ob[2]) - max(mb[0], ob[0])); ih = max(0, min(mb[3], ob[3]) - max(mb[1], ob[1]))
        inter = iw * ih
        box_iou.append(inter / ((mb[2] - mb[0]) * (mb[3] - mb[1]) + (ob[2] - ob[0]) * (ob[3] - ob[1]) - inter))
        mapping[f] = oid
        shutil.copyfile(mirror_dir / "images" / f, out / "images" / f"{oid}.jpg")
        shutil.copyfile(mirror_dir / "masks" / f"mask_{n}.jpg", out / "masks" / f"{oid}.jpg")
    box_iou = np.array(box_iou)
    assert size_ok == 1000, f"Solo {size_ok}/1000 tamanos coinciden con el json oficial"
    json.dump(mapping, open(out / "mirror_to_official_ids.json", "w"), indent=1)
    print(f"Restaurados 1000 nombres oficiales. Tamano exacto: {size_ok}/1000. IoU caja(mascara) vs caja oficial: "
          f"min={box_iou.min():.3f} p5={np.percentile(box_iou, 5):.3f} mediana={np.median(box_iou):.3f}")


def from_zip(zpath: Path, out: Path):
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out.parent)
    src = out.parent / "Kvasir-SEG"
    assert (src / "images").is_dir(), "El zip no tiene la estructura esperada Kvasir-SEG/images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official_zip", type=Path)
    ap.add_argument("--mirror_dir", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / "data" / "Kvasir-SEG"
    if a.official_zip:
        from_zip(a.official_zip, out)
        official = json.load(open(out / "kavsir_bboxes.json"))
    elif a.mirror_dir:
        # el json oficial de cajas (kavsir_bboxes.json del zip oficial) debe copiarse a data/ antes
        official = json.load(open(ROOT / "data" / "kavsir_bboxes_official.json"))
        from_mirror(a.mirror_dir, out, official)
    else:
        raise SystemExit("Indique --official_zip o --mirror_dir")
    ids = sorted(p.stem for p in (out / "images").glob("*.jpg"))
    assert len(ids) == 1000 and set(ids) == set(official), "Faltan imagenes o hay ids no oficiales"
    tr = [l.strip() for l in open(ROOT / "data/splits/official_mediaeval2020_train.txt") if l.strip()]
    te = [l.strip() for l in open(ROOT / "data/splits/official_mediaeval2020_val.txt") if l.strip()]
    assert len(tr) == 880 and len(te) == 120 and not set(tr) & set(te) and set(tr) | set(te) == set(ids)
    rng = random.Random(a.seed)
    tr_sh = tr[:]; rng.shuffle(tr_sh)
    dev, train = sorted(tr_sh[:100]), sorted(tr_sh[100:])
    for name, lst in [("train", train), ("dev", dev), ("test", sorted(te))]:
        (ROOT / "data/splits" / f"{name}.txt").write_text("\n".join(lst) + "\n")
    h = hashlib.md5("".join(sorted(te)).encode()).hexdigest()[:8]
    print(f"train={len(train)} dev={len(dev)} test={len(te)} (md5 ids test={h})")


if __name__ == "__main__":
    main()
