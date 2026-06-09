import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models.registry import build_model as build_registered_model, save_checkpoint
from training.experiment import append_metrics, create_run_dir, load_config, save_config, save_json, write_metrics_header
from training.utils import CIFAR10BatchDataset, get_device, set_seed


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, test_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train SimpleCNN on local CIFAR-10 batches and log a research run.")
    parser.add_argument("--config", type=Path, default=Path("configs/simple_cnn.yaml"), help="YAML experiment config.")
    parser.add_argument("--experiment-name", help="Override experiment name.")
    parser.add_argument("--epochs", type=int, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--lr", type=float, help="Override learning rate.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument("--early-stopping-patience", type=int, help="Stop after this many epochs without test_acc improvement. Use 0 to disable.")
    return parser.parse_args()


def apply_overrides(config, args):
    overrides = {
        "experiment_name": args.experiment_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "early_stopping_patience": args.early_stopping_patience,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def build_dataloaders(config, device):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = CIFAR10BatchDataset(config["data_dir"], train=True, transform=transform_train)
    test_dataset = CIFAR10BatchDataset(config["data_dir"], train=False, transform=transform_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    return train_loader, test_loader


def build_model(config):
    if config.get("dataset") != "cifar10":
        raise ValueError("training.train currently supports only dataset: cifar10")
    return build_registered_model(config.get("model"), num_classes=10)


def get_early_stopping_patience(config):
    patience = config.get("early_stopping_patience")
    if patience in (None, "", 0, "0"):
        return None
    patience = int(patience)
    if patience < 0:
        raise ValueError("early_stopping_patience must be >= 0")
    return patience or None


def main():
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    set_seed(int(config["seed"]))
    device = get_device()
    early_stopping_patience = get_early_stopping_patience(config)

    run_dir = create_run_dir(config["train_runs_dir"], config["experiment_name"])
    run_id = run_dir.name
    metrics_path = run_dir / "metrics.csv"
    best_checkpoint_path = run_dir / "checkpoint.pth"

    save_config(config, run_dir / "config.yaml")
    write_metrics_header(metrics_path)

    print(f"Run id: {run_id}")
    print(f"Run directory: {run_dir}")
    print(f"Best checkpoint path: {best_checkpoint_path}")
    print(f"Using device: {device}")
    if early_stopping_patience is None:
        print("Early stopping: disabled")
    else:
        print(f"Early stopping: patience={early_stopping_patience} epochs without test_acc improvement")

    train_loader, test_loader = build_dataloaders(config, device)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    best_acc = 0.0
    best_epoch = 0
    min_test_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False
    stop_reason = "completed_all_epochs"
    completed_epochs = 0
    run_start = time.perf_counter()

    for epoch in range(1, int(config["epochs"]) + 1):
        completed_epochs = epoch
        print(f"\nEpoch {epoch}/{config['epochs']}")
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        epoch_time = time.perf_counter() - epoch_start
        min_test_loss = min(min_test_loss, test_loss)

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.6f}",
            "test_loss": f"{test_loss:.6f}",
            "test_acc": f"{test_acc:.6f}",
            "epoch_time_sec": f"{epoch_time:.3f}",
        }
        append_metrics(metrics_path, row)

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Test loss : {test_loss:.4f} | Test acc : {test_acc:.4f}")
        print(f"Epoch time: {epoch_time:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_checkpoint_path,
                model,
                config,
                run_id,
                best_epoch=best_epoch,
                best_acc=best_acc,
                min_loss=min_test_loss,
            )
            print(f"Best checkpoint saved to: {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"No test_acc improvement for {epochs_without_improvement} epoch(s)")

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            stop_reason = f"no_test_acc_improvement_for_{early_stopping_patience}_epochs"
            print(f"Early stopping triggered: {stop_reason}")
            break

    total_time = time.perf_counter() - run_start
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "config": config,
        "epochs_requested": int(config["epochs"]),
        "epochs_completed": completed_epochs,
        "best_epoch": best_epoch,
        "best_test_acc": best_acc,
        "min_test_loss": min_test_loss if min_test_loss != float("inf") else None,
        "early_stopping_patience": early_stopping_patience,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "total_time_sec": total_time,
        "best_checkpoint_path": str(best_checkpoint_path),
    }
    save_json(summary, run_dir / "summary.json")
    print(f"\nBest test accuracy: {best_acc:.4f} at epoch {best_epoch}")
    print(f"Summary saved to: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()




