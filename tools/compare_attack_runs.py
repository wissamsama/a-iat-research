import argparse
import sys
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from project_paths import ATTACK_RUNS_DIR


def load_summary(run_dir):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def format_float(value):
    if value is None or value == "":
        return "-"
    return f"{float(value):.4f}"


def attack_date_time(run_name):
    parts = str(run_name).split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return str(run_name)


def model_file_name(config):
    model_path = config.get("model_path")
    if not model_path:
        return "-"
    file_name = Path(model_path).name
    parts = file_name.split("_", 2)
    if len(parts) == 3 and parts[0].count("-") == 2 and parts[1].count("-") == 2:
        return parts[2]
    return file_name


def print_column_help():
    lines = [
        "+-------------------- Colonnes compare_attack_runs --------------------+",
        "| date_time (run_attack) : date et heure du run d'attaque.         |",
        "| model     : fichier .pth attaque sans prefixe date/heure.        |",
        "| attack    : methode d'attaque utilisee, par exemple fgsm.         |",
        "| eps       : intensite maximale de perturbation en espace pixel.   |",
        "| dataset   : dataset utilise pour evaluer l'attaque.               |",
        "| samples   : nombre d'images evaluees par l'attaque.               |",
        "| clean     : accuracy sur les images originales.                   |",
        "| adv       : accuracy sur les images attaquees.                    |",
        "| drop      : baisse d'accuracy entre clean et adv.                 |",
        "| success   : part des predictions clean correctes devenues fausses.|",
        "+---------------------------------------------------------------------+",
    ]
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Compare attack runs saved in attack_runs/.")
    parser.add_argument("--attack-runs-dir", type=Path, default=ATTACK_RUNS_DIR)
    args = parser.parse_args()

    if not args.attack_runs_dir.exists():
        raise SystemExit(f"Attack runs folder not found: {args.attack_runs_dir}")

    rows = []
    for run_dir in sorted(path for path in args.attack_runs_dir.iterdir() if path.is_dir()):
        summary = load_summary(run_dir)
        if not summary:
            continue
        metrics = summary.get("metrics", {})
        config = summary.get("config", {})
        model = summary.get("model", {})
        run_name = summary.get("attack_run_id", run_dir.name)
        rows.append({
            "date_time": attack_date_time(run_name),
            "model_file": model_file_name(config),
            "attack": config.get("attack", "-"),
            "eps": config.get("epsilon", "-"),
            "dataset": model.get("dataset", "-"),
            "samples": metrics.get("num_samples", "-"),
            "clean_acc": metrics.get("clean_acc"),
            "adv_acc": metrics.get("adversarial_acc"),
            "drop": metrics.get("accuracy_drop"),
            "success": metrics.get("success_rate_on_clean_correct"),
        })

    if not rows:
        print("No attack runs found.")
        return

    print_column_help()
    print()
    rows.sort(key=lambda row: float(row["drop"] or 0.0), reverse=True)
    header = (
        f"{'date_time (run_attack)':22} | {'model':45} | {'attack':7} | {'eps':>7} | {'dataset':8} | "
        f"{'samples':>7} | {'clean':>8} | {'adv':>8} | {'drop':>8} | {'success':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['date_time'][:22]:22} | "
            f"{str(row['model_file'])[:45]:45} | "
            f"{str(row['attack'])[:7]:7} | "
            f"{str(row['eps']):>7} | "
            f"{str(row['dataset'])[:8]:8} | "
            f"{str(row['samples']):>7} | "
            f"{format_float(row['clean_acc']):>8} | "
            f"{format_float(row['adv_acc']):>8} | "
            f"{format_float(row['drop']):>8} | "
            f"{format_float(row['success']):>8}"
        )


if __name__ == "__main__":
    main()
