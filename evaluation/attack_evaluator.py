import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from attacks.fgsm import fgsm_attack
from tools.datasets import build_dataset, build_transforms


SUPPORTED_ATTACKS = {
    "fgsm": fgsm_attack,
}


def config_from_metadata(metadata, split):
    config = dict(metadata.get("config", {}))
    config["dataset"] = metadata.get("dataset", config.get("dataset", "cifar10"))
    config["num_classes"] = metadata.get("num_classes", config.get("num_classes"))
    config["normalization"] = metadata.get("normalization", config.get("normalization"))
    if metadata.get("input_shape"):
        config["input_size"] = int(metadata["input_shape"][1])
    config["split"] = split
    return config


def build_attack_dataloader(metadata, batch_size, split="test", max_samples=None):
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    config = config_from_metadata(metadata, split)
    transform = build_transforms(config, train=False)
    dataset = build_dataset(config, train=split == "train", transform=transform)
    if max_samples is not None:
        max_samples = min(int(max_samples), len(dataset))
        dataset = Subset(dataset, list(range(max_samples)))

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def _batch_stats(outputs, labels):
    loss = F.cross_entropy(outputs, labels, reduction="sum")
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return loss.item(), correct, preds


def evaluate_attack(model, metadata, attack_name, epsilon, batch_size=128, split="test", max_samples=None, device=None):
    if attack_name not in SUPPORTED_ATTACKS:
        available = ", ".join(sorted(SUPPORTED_ATTACKS))
        raise ValueError(f"Unsupported attack '{attack_name}'. Available attacks: {available}")

    device = device or next(model.parameters()).device
    loader = build_attack_dataloader(metadata, batch_size=batch_size, split=split, max_samples=max_samples)
    attack_fn = SUPPORTED_ATTACKS[attack_name]
    normalization = metadata["normalization"]

    totals = {
        "num_samples": 0,
        "clean_loss": 0.0,
        "clean_correct": 0,
        "adversarial_loss": 0.0,
        "adversarial_correct": 0,
        "attack_successes": 0,
    }
    start = time.perf_counter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_outputs = model(images)
            clean_loss, clean_correct, clean_preds = _batch_stats(clean_outputs, labels)

        adversarial_images = attack_fn(
            model,
            images,
            labels,
            epsilon=float(epsilon),
            mean=normalization["mean"],
            std=normalization["std"],
        )

        with torch.no_grad():
            adversarial_outputs = model(adversarial_images)
            adv_loss, adv_correct, adv_preds = _batch_stats(adversarial_outputs, labels)

        batch_size_current = labels.size(0)
        totals["num_samples"] += batch_size_current
        totals["clean_loss"] += clean_loss
        totals["clean_correct"] += clean_correct
        totals["adversarial_loss"] += adv_loss
        totals["adversarial_correct"] += adv_correct
        totals["attack_successes"] += ((clean_preds == labels) & (adv_preds != labels)).sum().item()

    elapsed = time.perf_counter() - start
    num_samples = totals["num_samples"]
    if num_samples == 0:
        raise ValueError("No samples evaluated.")

    clean_acc = totals["clean_correct"] / num_samples
    adversarial_acc = totals["adversarial_correct"] / num_samples
    clean_correct = max(totals["clean_correct"], 1)

    return {
        "attack": attack_name,
        "epsilon": float(epsilon),
        "split": split,
        "num_samples": num_samples,
        "clean_loss": totals["clean_loss"] / num_samples,
        "clean_acc": clean_acc,
        "adversarial_loss": totals["adversarial_loss"] / num_samples,
        "adversarial_acc": adversarial_acc,
        "accuracy_drop": clean_acc - adversarial_acc,
        "success_rate_on_clean_correct": totals["attack_successes"] / clean_correct,
        "duration_sec": elapsed,
    }

