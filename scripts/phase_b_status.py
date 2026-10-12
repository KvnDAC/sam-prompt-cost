"""Read experiment progress without importing Torch or interrupting the run."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True)
    a = ap.parse_args()
    status = json.loads((a.experiment / "status.json").read_text(encoding="utf-8"))
    try:
        meta = json.loads((ROOT / "cache/meta.json").read_text())
    except (OSError, json.JSONDecodeError):
        meta = {}
    counts = {}
    for split in ("dev", "train400"):
        ids = (ROOT / "data/splits" / f"{split}.txt").read_text().split()
        ready = sum(i in meta and meta[i].get("precision") == "fp32" and meta[i].get("storage") == "fp32"
                    and (ROOT / "cache/emb" / f"{i}.npy").is_file() for i in ids)
        counts[split] = {"ready": ready, "total": len(ids)}
    current = status.get("current_stage")
    stage = status["stages"].get(current, {})
    log = Path(stage["log"]) if "log" in stage else None
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-3:] if log and log.is_file() else []
    print(json.dumps({"state": status["state"], "current_stage": current,
                      "embeddings": counts,
                      "disk_free_gib": round(shutil.disk_usage(ROOT).free / 1024**3, 2),
                      "completed_stages": [k for k, v in status["stages"].items() if v["state"] == "complete"],
                      "last_log_lines": tail}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
