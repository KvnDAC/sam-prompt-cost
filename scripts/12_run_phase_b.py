"""Run the complete development experiment, sequentially, without evaluating test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from samcost.common import save_json


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_root", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--wait_for_dev_cache", action="store_true", help="esperar al encoder dev que ya está en ejecución")
    a = ap.parse_args()
    root = a.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    status_path = root / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"created_utc": now(), "stages": {}}
    env = {**os.environ, "SAMCOST_THREADS": str(a.threads), "PYTHONIOENCODING": "utf-8"}
    grid = [1e-5, 3e-5, 1e-4]
    plan = {"train_split": "train400", "n_train": 400, "n_dev": 100,
               "evaluation_split": "dev", "test_evaluated": False, "storage": "fp32",
               "encoder_precision": "fp32", "intermediate_blocks": False,
               "seeds": [0, 1, 2], "k": 4, "init": "meanpoint", "include_initial": True,
               "lr_grid_conditional": grid, "search_epochs": 3, "final_epochs": 6,
               "static_lr": 0.005, "batch_size": 8, "threads": a.threads,
               "selection": "best mean dev Dice, ties prefer smaller LR",
               "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [ROOT / "scripts/02_encode.py", ROOT / "scripts/06_probe.py",
                                            ROOT / "scripts/07_learned_prompts.py", ROOT / "scripts/10_prompt_perturbations.py",
                                            ROOT / "src/samcost/learned.py", ROOT / "src/samcost/decode.py",
                                            ROOT / "src/samcost/data.py"]},
               "note": "Development scores are used for selection, not held-out generalization."}
    plan_path = root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("La configuración o el código cambiaron; use una carpeta nueva de experimento.")
    save_json(plan, plan_path)

    if a.wait_for_dev_cache:
        from samcost.common import CACHE
        from samcost.data import read_split
        dev_ids = read_split("dev")
        while True:
            try:
                meta = json.loads((CACHE / "meta.json").read_text())
            except (OSError, json.JSONDecodeError):
                meta = {}
            count = sum(i in meta and meta[i].get("precision") == "fp32" and meta[i].get("storage") == "fp32"
                        and (CACHE / "emb" / f"{i}.npy").is_file() for i in dev_ids)
            if count == len(dev_ids):
                break
            status.update(state="running", current_stage="await_dev_cache", dev_embeddings_ready=count)
            save_json(status, status_path)
            time.sleep(3)

    def run(name, script, *args):
        if status["stages"].get(name, {}).get("state") == "complete":
            return
        command = [sys.executable, str(ROOT / "scripts" / script), *map(str, args)]
        log = logs / f"{name}.log"
        status["current_stage"] = name
        status["state"] = "running"
        status["stages"][name] = {"state": "running", "started_utc": now(), "command": command, "log": str(log)}
        save_json(status, status_path)
        print(f"START {name}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        stage = status["stages"][name]
        stage.update(state="complete" if result.returncode == 0 else "failed", ended_utc=now(), exit_code=result.returncode)
        save_json(status, status_path)
        if result.returncode:
            status["state"] = "failed"
            save_json(status, status_path)
            raise RuntimeError(f"{name} failed; inspect {log}")
        print(f"END {name}", flush=True)

    run("encode_dev", "02_encode.py", "--split", "dev", "--precision", "fp32", "--storage", "fp32", "--no_blocks")
    run("perturbations_dev", "10_prompt_perturbations.py", "--split", "dev", "--output_dir", root / "perturbations")
    run("encode_train400", "02_encode.py", "--split", "train400", "--precision", "fp32", "--storage", "fp32", "--no_blocks")

    shared = ["--k", 4, "--init", "meanpoint", "--bs", 8, "--dev_only", "--include_initial"]
    searches = []
    for lr in grid:
        tag = f"cp_k4_s0_lr{lr:g}"
        run(f"search_{tag}", "07_learned_prompts.py", *shared, "--model", "conditional", "--seed", 0,
            "--epochs", 3, "--lr", lr, "--tag", tag, "--output_root", root / "search")
        summary = json.loads((root / "search" / tag / "summary.json").read_text())
        searches.append({"lr": lr, "dev_dice": summary["best_dev_dice"], "best_epoch": summary["best_epoch"], "tag": tag})
    chosen = max(searches, key=lambda x: (x["dev_dice"], -x["lr"]))
    save_json({"chosen": chosen, "candidates": searches, "selected_utc": now(), "test_evaluated": False}, root / "selection.json")

    for model, lr in (("static", 0.005), ("conditional", chosen["lr"])):
        for seed in (0, 1, 2):
            tag = f"{model}_k4_s{seed}"
            run(tag, "07_learned_prompts.py", *shared, "--model", model, "--seed", seed,
                "--epochs", 6, "--lr", lr, "--tag", tag, "--output_root", root / "runs")
    for seed in (0, 1, 2):
        run(f"probe_s{seed}", "06_probe.py", "--part", "main", "--split", "dev", "--seed", seed,
            "--output_root", root / "probes" / f"seed{seed}")
    run("summarize", "13_summarize_phase_b.py", "--experiment", root)
    status.update(state="complete", current_stage=None, ended_utc=now())
    save_json(status, status_path)
    print(f"Development experiment complete: {root / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
