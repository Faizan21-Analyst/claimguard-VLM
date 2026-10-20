"""Score vision backends on the golden set with the SAME scoring code as training/colab_eval_base.py.

    python -m scripts.eval_golden --backend remote_ft --limit 5          # smoke test: 5 photos
    python -m scripts.eval_golden --backend remote_ft --backend groq     # full run, both, then a comparison table
    python -m scripts.eval_golden --compare --also results/base_golden.json   # table only, no model calls

What it checks: that the remote GPU service reproduces your Colab result, and how the fine-tuned model compares
with Groq on the same photos. Valid-JSON rate and the per-field accuracy come from compute_metrics(), unchanged.

Safe to stop and run again:
  * every answer is saved the moment it arrives (results/golden_<backend>.jsonl) and a rerun skips finished photos
  * if the GPU service says it is unavailable (daily quota, queue, timeout) the run STOPS. Those photos are not
    scored as wrong answers, because a missing answer is not evidence about the model. Run the same command
    tomorrow and it continues where it stopped.

A named backend is used exactly as named: there is never a fallback in this script.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from colab_eval_base import compute_metrics, parse_prediction  # noqa: E402  (the Colab scoring, unchanged)

from claimguard.vlm.base import ModelUnavailable  # noqa: E402
from claimguard.vlm.factory import get_vlm  # noqa: E402

GOLDEN = ROOT / "data" / "golden" / "golden.jsonl"
RESULTS = ROOT / "results"

TABLE_ROWS = [
    ("photos scored", "n"),
    ("valid JSON", "valid_json_rate"),
    ("damage_type accuracy", "damage_type_acc"),
    ("damage_type macro-F1", "damage_type_macro_f1"),
    ("damage_type acc, real photos", "damage_type_acc_real"),
    ("damage_type acc, poor photos", "damage_type_acc_synthetic_poor"),
    ("severity accuracy", "severity_acc"),
    ("component accuracy", "component_acc"),
    ("image_quality accuracy", "image_quality_acc"),
    ("seconds per photo", "seconds_per_image"),
]


def load_rows(limit: int) -> list[dict]:
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def read_progress(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    entries = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {e["id"]: e for e in entries}


def run_backend(name: str, rows: list[dict], restart: bool) -> dict:
    RESULTS.mkdir(exist_ok=True)
    progress_file = RESULTS / f"golden_{name}.jsonl"
    if restart and progress_file.exists():
        progress_file.unlink()

    vlm = get_vlm(name)
    done = read_progress(progress_file)
    todo = [r for r in rows if r["id"] not in done]
    print(f"\n== {name}  ({vlm.name}): {len(done)} done, {len(todo)} to do")
    if name == "remote_ft" and todo:
        print(f"   A free ZeroGPU quota is a few minutes a day; {len(todo)} photos may use most of it. "
              "Progress is saved, so you can stop and continue later.")

    stopped = None
    with progress_file.open("a", encoding="utf-8") as out:
        for number, row in enumerate(todo, start=1):
            started = time.time()
            try:
                raw = vlm._raw(GOLDEN.parent / row["image"])
            except ModelUnavailable as exc:
                stopped = str(exc)
                break
            except ValueError:
                raw = ""   # empty or cut-off output: counts as invalid JSON, like in the Colab runs
            entry = {"id": row["id"], "raw": raw, "pred": parse_prediction(raw), "seconds": round(time.time() - started, 2)}
            out.write(json.dumps(entry) + "\n")
            out.flush()
            done[row["id"]] = entry
            if number % 10 == 0:
                print(f"   {number}/{len(todo)}")

    scored = [r for r in rows if r["id"] in done]
    summary = {"model": vlm.name, "n_total": len(rows), "n_done": len(scored), "complete": len(scored) == len(rows)}
    if scored:
        metrics = compute_metrics(scored, [done[r["id"]]["pred"] for r in scored])
        metrics["seconds_per_image"] = sum(done[r["id"]]["seconds"] for r in scored) / len(scored)
        summary["metrics"] = metrics
    (RESULTS / f"golden_{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if stopped:
        print(f"   STOPPED: {stopped} {len(scored)}/{len(rows)} photos are scored. Run the same command later to continue.")
    return summary


def cell(summary: dict | None, key: str) -> str:
    value = ((summary or {}).get("metrics") or {}).get(key)
    if value is None:
        return "-"
    return str(int(value)) if key == "n" else (f"{value:.1f}" if key == "seconds_per_image" else f"{value:.3f}")


def print_table(columns: dict[str, dict]) -> None:
    names = list(columns)
    width = max(len(label) for label, _ in TABLE_ROWS) + 2
    print("\n" + " " * width + "".join(f"{n[:26]:>28}" for n in names))
    for label, key in TABLE_ROWS:
        print(f"{label:<{width}}" + "".join(f"{cell(columns[n], key):>28}" for n in names))
    for n in names:
        s = columns[n]
        if s and not s.get("complete", True):
            print(f"\n! {n}: only {s.get('n_done')} of {s.get('n_total')} photos scored so far; compare with care.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", action="append", help="remote_ft, groq, local_ft, local_base (repeatable)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N golden photos (smoke test)")
    ap.add_argument("--restart", action="store_true", help="discard saved progress for the chosen backends")
    ap.add_argument("--compare", action="store_true", help="only print the table from saved results")
    ap.add_argument("--also", type=Path, action="append", default=[],
                    help="a results file from the Colab scripts to include, e.g. results/base_golden.json")
    args = ap.parse_args()

    backends = args.backend or ["remote_ft"]
    columns: dict[str, dict] = {}

    if args.compare:
        for path in sorted(RESULTS.glob("golden_*.json")):
            columns[path.stem.removeprefix("golden_")] = json.loads(path.read_text(encoding="utf-8"))
    else:
        rows = load_rows(args.limit)
        for name in backends:
            columns[name] = run_backend(name, rows, args.restart)

    for extra in args.also:
        data = json.loads(extra.read_text(encoding="utf-8"))
        columns[f"colab:{extra.stem}"] = {"metrics": data["metrics"], "complete": True}

    if columns:
        print_table(columns)
    else:
        print("Nothing to show yet. Run with --backend first.")


if __name__ == "__main__":
    main()
