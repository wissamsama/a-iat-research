import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from evaluation.attack_evaluator import SUPPORTED_ATTACKS, evaluate_attack
from models.loading import load_trained_model
from project_paths import ATTACK_RUNS_DIR, CONFIGS_DIR, TRAINED_MODELS_DIR, as_project_path, safe_name, timestamp_id
from training.utils import get_device, set_seed

DEFAULT_CONFIG = CONFIGS_DIR / "fgsm.yaml"
DEFAULT_TRAINED_MODELS_DIR = TRAINED_MODELS_DIR
DEFAULT_ATTACK_RUNS_DIR = ATTACK_RUNS_DIR


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def apply_overrides(config, args):
    overrides = {
        "attack": args.attack,
        "epsilon": args.epsilon,
        "batch_size": args.batch_size,
        "split": args.split,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "model_path": str(args.model) if args.model else None,
        "trained_models_dir": str(args.trained_models_dir) if args.trained_models_dir else None,
        "attack_runs_dir": str(args.attack_runs_dir) if args.attack_runs_dir else None,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def attack_run_id(attack, metadata):
    model_name = str(metadata.get("model_name", "model")).replace("_", "")
    dataset = metadata.get("dataset", "dataset")
    return safe_name(f"{timestamp_id()}_{attack}_{model_name}_{dataset}")


def available_models(models_dir):
    return sorted(path for path in models_dir.glob("*.pth") if path.is_file())


def choose_model(models_dir):
    models = available_models(models_dir)
    if not models:
        raise SystemExit(f"No promoted model found in: {models_dir}")

    print("\nModeles disponibles:")
    for index, model_path in enumerate(models, start=1):
        print(f"  {index}. {model_path.name}")

    while True:
        answer = input("\nModele a attaquer: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(models):
            return models[int(answer) - 1]
        print("Choix invalide.")


def source_checkpoint_path(model_path, metadata):
    source = metadata.get("source_checkpoint_path")
    if source:
        return source
    if metadata.get("artifact_type") == "run_checkpoint":
        return str(model_path)
    return None


def write_metrics(path, metrics):
    fields = [
        "attack",
        "epsilon",
        "split",
        "num_samples",
        "clean_loss",
        "clean_acc",
        "adversarial_loss",
        "adversarial_acc",
        "accuracy_drop",
        "success_rate_on_clean_correct",
        "duration_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: metrics[field] for field in fields})


def parse_args():
    parser = argparse.ArgumentParser(description="Run an adversarial attack against a trained model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML attack config.")
    parser.add_argument("--model", type=Path, help="Override path to a promoted model or train checkpoint .pth.")
    parser.add_argument("--trained-models-dir", type=Path, help="Override folder containing promoted models.")
    parser.add_argument("--attack-runs-dir", type=Path, help="Override folder where attack runs are written.")
    parser.add_argument("--attack", choices=sorted(SUPPORTED_ATTACKS), help="Override attack name.")
    parser.add_argument("--epsilon", type=float, help="Override perturbation budget in pixel space [0, 1].")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--split", choices=["train", "test"], help="Override split.")
    parser.add_argument("--max-samples", type=int, help="Override sample limit. Use config value if omitted.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)

    set_seed(int(config.get("seed", 42)))
    device = get_device()

    trained_models_dir = as_project_path(config.get("trained_models_dir"), DEFAULT_TRAINED_MODELS_DIR)
    attack_runs_dir = as_project_path(config.get("attack_runs_dir"), DEFAULT_ATTACK_RUNS_DIR)
    model_path = as_project_path(config.get("model_path")) if config.get("model_path") else choose_model(trained_models_dir)

    model, metadata = load_trained_model(model_path, map_location=device)

    metrics = evaluate_attack(
        model,
        metadata,
        attack_name=config.get("attack", "fgsm"),
        epsilon=float(config.get("epsilon", 0.03)),
        batch_size=int(config.get("batch_size", 128)),
        split=config.get("split", "test"),
        max_samples=config.get("max_samples"),
        device=device,
    )

    run_id = attack_run_id(config.get("attack", "fgsm"), metadata)
    run_dir = attack_runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    saved_config = {
        "attack": config.get("attack", "fgsm"),
        "model_path": str(model_path),
        "source_checkpoint_path": source_checkpoint_path(model_path, metadata),
        "trained_models_dir": str(trained_models_dir),
        "attack_runs_dir": str(attack_runs_dir),
        "split": config.get("split", "test"),
        "epsilon": float(config.get("epsilon", 0.03)),
        "batch_size": int(config.get("batch_size", 128)),
        "max_samples": config.get("max_samples"),
        "seed": int(config.get("seed", 42)),
        "device": str(device),
    }
    summary = {
        "attack_run_id": run_id,
        "attack_run_dir": str(run_dir),
        "device": str(device),
        "config": saved_config,
        "model": {
            "model_name": metadata.get("model_name"),
            "dataset": metadata.get("dataset"),
            "num_classes": metadata.get("num_classes"),
            "source_run_id": metadata.get("source_summary", {}).get("run_id", metadata.get("run_id")),
        },
        "metrics": metrics,
    }

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(saved_config, file, sort_keys=False)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    write_metrics(run_dir / "metrics.csv", metrics)

    print(f"Attack run: {run_id}")
    print(f"Attack run directory: {run_dir}")
    print(f"Clean acc: {metrics['clean_acc']:.4f}")
    print(f"Adversarial acc: {metrics['adversarial_acc']:.4f}")
    print(f"Accuracy drop: {metrics['accuracy_drop']:.4f}")


if __name__ == "__main__":
    main()