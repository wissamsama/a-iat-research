import argparse
import sys
import csv
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from project_paths import TRAIN_RUNS_DIR


def load_summary(run_dir):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def load_metrics(run_dir):
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return []
    with metrics_path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def min_metric(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return min(values) if values else None


def format_float(value):
    if value is None or value == "":
        return "-"
    return f"{float(value):.4f}"


def file_status(path):
    if not path:
        return "-"
    return "ok" if Path(path).exists() else "missing"


def best_artifact_path(summary):
    if not summary:
        return None
    return summary.get("best_checkpoint_path")


def completed_epochs(summary):
    if not summary:
        return "-"
    return summary.get("epochs_completed", summary.get("epochs", "-"))


def main():
    parser = argparse.ArgumentParser(description="Compare training runs saved in train_runs/.")
    parser.add_argument("--train-runs-dir", type=Path, default=TRAIN_RUNS_DIR, help="Folder containing train run directories.")
    args = parser.parse_args()

    if not args.train_runs_dir.exists():
        raise SystemExit(f"Train runs folder not found: {args.train_runs_dir}")

    rows = []
    for run_dir in sorted(path for path in args.train_runs_dir.iterdir() if path.is_dir()):
        summary = load_summary(run_dir)
        metrics = load_metrics(run_dir)
        if summary is None and not metrics:
            continue

        config = summary.get("config", {}) if summary else {}
        artifact_path = best_artifact_path(summary)
        mode = config.get("training_mode", summary.get("training_mode", "classic"))
        rows.append({
            "run": summary.get("run_id", run_dir.name) if summary else run_dir.name,
            "model": config.get("model", "-"),
            "mode": mode,
            "lambda": config.get("clean_loss_lambda", summary.get("clean_loss_lambda", "-")) if mode == "adversarial" else "-",
            "adv_eps": config.get("adv_training_epsilon", summary.get("adv_training_epsilon", "-")) if mode == "adversarial" else "-",
            "trained_ep": completed_epochs(summary),
            "best_ep": summary.get("best_epoch", "-") if summary else "-",
            "lr": config.get("learning_rate", "-"),
            "batch": config.get("batch_size", "-"),
            "best_acc": summary.get("best_test_acc", None) if summary else None,
            "min_loss": summary.get("min_test_loss", min_metric(metrics, "test_loss")) if summary else min_metric(metrics, "test_loss"),
            "time_sec": summary.get("total_time_sec", None) if summary else None,
            "checkpoint": file_status(artifact_path),
        })

    if not rows:
        print("No completed train runs found.")
        return

    rows.sort(key=lambda row: float(row["best_acc"] or 0.0), reverse=True)
    header = (
        f"{'run':45} | {'model':12} | {'mode':11} | {'lambda':>6} | {'adv_eps':>7} | "
        f"{'trained':>7} | {'best_ep':>7} | {'lr':>10} | {'batch':>7} | "
        f"{'best_acc':>8} | {'min_loss':>8} | {'time(s)':>9} | {'checkpoint':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['run'][:45]:45} | "
            f"{str(row['model'])[:12]:12} | "
            f"{str(row['mode'])[:11]:11} | "
            f"{str(row['lambda']):>6} | "
            f"{str(row['adv_eps']):>7} | "
            f"{str(row['trained_ep']):>7} | "
            f"{str(row['best_ep']):>7} | "
            f"{str(row['lr']):>10} | "
            f"{str(row['batch']):>7} | "
            f"{format_float(row['best_acc']):>8} | "
            f"{format_float(row['min_loss']):>8} | "
            f"{format_float(row['time_sec']):>9} | "
            f"{row['checkpoint']:>10}"
        )


if __name__ == "__main__":
    main()




