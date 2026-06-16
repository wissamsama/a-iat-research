from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = PROJECT_DIR / "configs"
DATA_DIR = PROJECT_DIR / "data"
TRAIN_RUNS_DIR = PROJECT_DIR / "train_runs"
ATTACK_RUNS_DIR = PROJECT_DIR / "attack_runs"
TRAINED_MODELS_DIR = PROJECT_DIR / "trained_models"


def as_project_path(value, default=None):
    if value is None:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def project_relative(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def safe_name(value):
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))


def timestamp_id():
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")