import torch

from models.simple_cnn import SimpleCNN
from tools.datasets import (
    class_names_for_dataset,
    data_dir_from_config,
    dataset_name_from_config,
    input_shape_for_dataset,
    normalization_for_dataset,
    num_classes_for_dataset,
)


MODEL_BUILDERS = {
    "simple_cnn": SimpleCNN,
}


def build_model(model_name, num_classes, input_size=32):
    try:
        builder = MODEL_BUILDERS[model_name]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(f"Unsupported model '{model_name}'. Available models: {available}") from error
    return builder(num_classes=int(num_classes), input_size=int(input_size))


def infer_num_classes(dataset, config=None):
    return num_classes_for_dataset(dataset, config)


def checkpoint_metadata(config, run_id, best_epoch=None, best_acc=None, min_loss=None):
    dataset = dataset_name_from_config(config)
    model_name = str(config.get("model", "simple_cnn"))
    num_classes = num_classes_for_dataset(dataset, config)
    data_dir = data_dir_from_config(config)
    return {
        "artifact_type": "run_checkpoint",
        "format_version": 1,
        "run_id": run_id,
        "model_name": model_name,
        "dataset": dataset,
        "num_classes": num_classes,
        "class_names": class_names_for_dataset(dataset, data_dir=data_dir),
        "input_shape": input_shape_for_dataset(dataset, config),
        "normalization": normalization_for_dataset(dataset, config),
        "preprocessing": {
            "expected_input_range": "0_1_before_normalization",
            "output": "logits",
        },
        "config": dict(config),
        "metrics": {
            "best_epoch": best_epoch,
            "best_acc": best_acc,
            "min_loss": min_loss,
        },
    }


def save_checkpoint(path, model, config, run_id, best_epoch=None, best_acc=None, min_loss=None):
    checkpoint = checkpoint_metadata(config, run_id, best_epoch, best_acc, min_loss)
    checkpoint["model_state_dict"] = model.state_dict()
    torch.save(checkpoint, path)


def is_enriched_checkpoint(data):
    return isinstance(data, dict) and "model_state_dict" in data and "model_name" in data

