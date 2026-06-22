import csv
import json
from datetime import datetime
from pathlib import Path

import yaml


METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "train_acc",
    "train_clean_loss",
    "train_adv_loss",
    "train_adv_acc",
    "test_loss",
    "test_acc",
    "epoch_time_sec",
]


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_config(config, path):
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def save_json(data, path):
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def create_run_dir(train_runs_dir, experiment_name):
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in experiment_name)
    run_dir = Path(train_runs_dir) / f"{timestamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_metrics_header(path):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()


def append_metrics(path, row):
    with Path(path).open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writerow(row)

