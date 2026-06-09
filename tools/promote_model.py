import argparse
import json
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_RUNS_DIR = PROJECT_DIR / "train_runs"
DEFAULT_TRAINED_MODELS_DIR = PROJECT_DIR / "trained_models"


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Not an enriched checkpoint: {path}")
    return checkpoint


def load_summary_for_checkpoint(checkpoint_path):
    summary_path = checkpoint_path.parent / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def project_relative(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def normalized_checkpoint_id(path):
    path = Path(path)
    candidates = []
    if not path.is_absolute():
        candidates.append((PROJECT_DIR / path).resolve())
    if path.exists():
        candidates.append(path.resolve())
    candidates.append(Path(project_relative(path)))
    return {candidate.as_posix().lower() for candidate in candidates}


def metric_value(checkpoint, summary, summary_key, checkpoint_key):
    if summary_key in summary:
        return summary[summary_key]
    return checkpoint.get("metrics", {}).get(checkpoint_key)


def promoted_name(checkpoint, summary):
    best_acc = metric_value(checkpoint, summary, "best_test_acc", "best_acc")
    acc_part = f"_acc{float(best_acc):.4f}" if best_acc is not None else ""
    run_id = summary.get("run_id", checkpoint.get("run_id", "unknown_run"))
    date_part = "_".join(str(run_id).split("_")[:2])
    return f"{date_part}_{checkpoint['model_name']}_{checkpoint['dataset']}{acc_part}.pth"


def already_promoted(checkpoint_path, trained_models_dir):
    checkpoint_ids = normalized_checkpoint_id(checkpoint_path)
    for model_path in trained_models_dir.glob("*.pth"):
        try:
            artifact = torch.load(model_path, map_location="cpu")
        except Exception:
            continue
        if not isinstance(artifact, dict):
            continue
        source_path = artifact.get("source_checkpoint_path")
        if source_path and checkpoint_ids.intersection(normalized_checkpoint_id(source_path)):
            return True
    return False


def find_promotable_checkpoints(train_runs_dir, trained_models_dir):
    items = []
    for checkpoint_path in sorted(train_runs_dir.glob("*/checkpoint.pth")):
        checkpoint = load_checkpoint(checkpoint_path)
        summary = load_summary_for_checkpoint(checkpoint_path)
        output_path = trained_models_dir / promoted_name(checkpoint, summary)
        if not already_promoted(checkpoint_path, trained_models_dir):
            items.append((checkpoint_path, checkpoint, summary, output_path))
    return items


def choose_checkpoint(items):
    print("\nCheckpoints non promus:")
    for index, (path, checkpoint, summary, output_path) in enumerate(items, start=1):
        best_acc = metric_value(checkpoint, summary, "best_test_acc", "best_acc")
        best_epoch = metric_value(checkpoint, summary, "best_epoch", "best_epoch")
        trained_epochs = summary.get("epochs_completed", "-")
        run_id = summary.get("run_id", checkpoint.get("run_id", path.parent.name))
        print(
            f"  {index}. {run_id} | {checkpoint['model_name']} | {checkpoint['dataset']} | "
            f"acc={best_acc} | best_ep={best_epoch} | trained_ep={trained_epochs}"
        )
        print(f"     -> {output_path.name}")

    while True:
        try:
            answer = input("\nModele a promouvoir: ").strip()
        except EOFError:
            print("Input closed.")
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(items):
            return items[int(answer) - 1]
        print("Choix invalide.")


def ask_output_path(default_path):
    print(f"\nNom par defaut: {default_path.name}")
    answer = input("Nouveau nom (laisser vide pour garder le nom par defaut): ").strip()
    if not answer:
        return default_path
    output_name = answer if answer.endswith(".pth") else f"{answer}.pth"
    return default_path.parent / output_name


def promote(checkpoint_path, checkpoint, summary, output_path, force=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError(f"Trained model already exists: {output_path}")

    artifact = dict(checkpoint)
    artifact["artifact_type"] = "trained_model"
    artifact["source_checkpoint_path"] = project_relative(checkpoint_path)
    artifact["source_summary"] = summary
    torch.save(artifact, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Promote an enriched run checkpoint into trained_models/.")
    parser.add_argument("--train-runs-dir", type=Path, default=DEFAULT_TRAIN_RUNS_DIR)
    parser.add_argument("--trained-models-dir", type=Path, default=DEFAULT_TRAINED_MODELS_DIR)
    parser.add_argument("--checkpoint", type=Path, help="Specific run checkpoint to promote.")
    parser.add_argument("--name", help="Output filename inside trained_models/. Defaults to a generated research name.")
    parser.add_argument("--force", action="store_true", help="Overwrite the output file if it already exists.")
    args = parser.parse_args()

    if args.checkpoint:
        checkpoint_path = args.checkpoint
        checkpoint = load_checkpoint(checkpoint_path)
        summary = load_summary_for_checkpoint(checkpoint_path)
        output_name = args.name or promoted_name(checkpoint, summary)
        output_path = args.trained_models_dir / output_name
        promoted_path = promote(checkpoint_path, checkpoint, summary, output_path, force=args.force)
        print(f"Promoted model: {promoted_path}")
        return

    while True:
        items = find_promotable_checkpoints(args.train_runs_dir, args.trained_models_dir)
        if not items:
            print("No unpromoted checkpoints found.")
            return
        selected = choose_checkpoint(items)
        if selected is None:
            return
        checkpoint_path, checkpoint, summary, output_path = selected
        output_path = ask_output_path(output_path)
        promoted_path = promote(checkpoint_path, checkpoint, summary, output_path, force=args.force)
        print(f"Promoted model: {promoted_path}")
        print("Ctrl+C pour fermer, ou continue pour promouvoir un autre modele.")


if __name__ == "__main__":
    main()


