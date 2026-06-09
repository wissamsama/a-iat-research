import torch

from models.simple_cnn import SimpleCNN


MODEL_BUILDERS = {
    "simple_cnn": SimpleCNN,
}


def build_model(model_name, num_classes):
    try:
        builder = MODEL_BUILDERS[model_name]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(f"Unsupported model '{model_name}'. Available models: {available}") from error
    return builder(num_classes=int(num_classes))


def infer_num_classes(dataset, config=None):
    dataset = str(dataset).lower()
    if config and config.get("num_classes"):
        return int(config["num_classes"])
    if dataset == "cifar10":
        return 10
    if dataset == "cifar100":
        return 100
    if dataset == "gtsrb":
        return 43
    raise ValueError(f"Cannot infer num_classes for dataset '{dataset}'")


def class_names_for_dataset(dataset, num_classes):
    dataset = str(dataset).lower()
    if dataset == "cifar10":
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    return [f"class_{class_id:05d}" for class_id in range(int(num_classes))]


def checkpoint_metadata(config, run_id, best_epoch=None, best_acc=None, min_loss=None):
    dataset = str(config.get("dataset", "cifar10")).lower()
    model_name = str(config.get("model", "simple_cnn"))
    num_classes = infer_num_classes(dataset, config)
    return {
        "artifact_type": "run_checkpoint",
        "format_version": 1,
        "run_id": run_id,
        "model_name": model_name,
        "dataset": dataset,
        "num_classes": num_classes,
        "class_names": class_names_for_dataset(dataset, num_classes),
        "input_shape": [3, 32, 32],
        "normalization": {
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2470, 0.2435, 0.2616],
        },
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
