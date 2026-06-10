import csv
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DATASET_ALIASES = {
    "gtrsb": "gtsrb",
}

DATASET_SPECS = {
    "cifar10": {
        "display_name": "CIFAR-10",
        "folder": "CIFAR-10",
        "kind": "cifar_batch",
        "num_classes": 10,
        "input_shape": [3, 32, 32],
        "normalization": {
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2470, 0.2435, 0.2616],
        },
        "meta_file": "batches.meta",
        "label_names_key": "label_names",
        "label_field": "labels",
        "train_files": [f"data_batch_{index}" for index in range(1, 6)],
        "test_files": ["test_batch"],
        "train_augmentation": "cifar_32",
    },
    "cifar100": {
        "display_name": "CIFAR-100",
        "folder": "CIFAR-100",
        "kind": "cifar_batch",
        "num_classes": 100,
        "input_shape": [3, 32, 32],
        "normalization": {
            "mean": [0.5071, 0.4867, 0.4408],
            "std": [0.2675, 0.2565, 0.2761],
        },
        "meta_file": "meta",
        "label_names_key": "fine_label_names",
        "label_field": "fine_labels",
        "train_files": ["train"],
        "test_files": ["test"],
        "train_augmentation": "cifar_32",
    },
    "gtsrb": {
        "display_name": "GTSRB",
        "folder": "GTSRB",
        "kind": "gtsrb_folder",
        "num_classes": 43,
        "input_shape": [3, 32, 32],
        "normalization": {
            "mean": [0.3403, 0.3121, 0.3214],
            "std": [0.2724, 0.2608, 0.2669],
        },
        "train_augmentation": "resize_only",
    },
}


def normalize_dataset_name(dataset):
    name = str(dataset).lower()
    return DATASET_ALIASES.get(name, name)


def dataset_spec(dataset):
    name = normalize_dataset_name(dataset)
    try:
        return DATASET_SPECS[name]
    except KeyError as error:
        available = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unsupported dataset '{dataset}'. Available datasets: {available}") from error


def dataset_name_from_config(config):
    return normalize_dataset_name(config.get("dataset", "cifar10"))


def data_dir_from_config(config):
    dataset_name = dataset_name_from_config(config)
    if config.get("data_dir"):
        return Path(config["data_dir"])
    return Path("data") / dataset_spec(dataset_name)["folder"]


def num_classes_for_dataset(dataset, config=None):
    if config and config.get("num_classes"):
        return int(config["num_classes"])
    return int(dataset_spec(dataset)["num_classes"])


def input_shape_for_dataset(dataset, config=None):
    spec = dataset_spec(dataset)
    if config and config.get("input_size"):
        input_size = int(config["input_size"])
        return [3, input_size, input_size]
    return list(spec["input_shape"])


def normalization_for_dataset(dataset, config=None):
    if config and config.get("normalization"):
        normalization = config["normalization"]
        return {
            "mean": list(normalization["mean"]),
            "std": list(normalization["std"]),
        }
    spec = dataset_spec(dataset)
    return {
        "mean": list(spec["normalization"]["mean"]),
        "std": list(spec["normalization"]["std"]),
    }


def class_names_for_dataset(dataset, data_dir=None):
    spec = dataset_spec(dataset)
    if spec["kind"] == "cifar_batch" and data_dir is not None:
        metadata_path = Path(data_dir) / spec["meta_file"]
        if metadata_path.exists():
            metadata = load_pickle(metadata_path)
            return list(metadata[spec["label_names_key"]])
    if normalize_dataset_name(dataset) == "cifar10":
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
    return [f"class_{class_id:05d}" for class_id in range(int(spec["num_classes"]))]


def load_pickle(path):
    with Path(path).open("rb") as file:
        return pickle.load(file, encoding="latin1")


class CifarBatchDataset(Dataset):
    def __init__(self, data_dir, dataset, train=True, transform=None):
        self.data_dir = Path(data_dir)
        self.dataset = normalize_dataset_name(dataset)
        self.spec = dataset_spec(self.dataset)
        self.transform = transform

        if not self.data_dir.exists():
            raise FileNotFoundError(f"{self.spec['display_name']} data folder not found: {self.data_dir}")

        batch_names = self.spec["train_files"] if train else self.spec["test_files"]
        images = []
        labels = []
        for batch_name in batch_names:
            batch_path = self.data_dir / batch_name
            if not batch_path.exists():
                raise FileNotFoundError(f"Missing {self.spec['display_name']} batch file: {batch_path}")
            batch = load_pickle(batch_path)
            batch_images = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            images.append(batch_images)
            labels.extend(batch[self.spec["label_field"]])

        self.images = np.concatenate(images).astype(np.uint8)
        self.labels = np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        label = int(self.labels[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class GTSRBDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = self._load_train_samples() if train else self._load_test_samples()
        if not self.samples:
            split_name = "train" if train else "test"
            raise FileNotFoundError(f"No GTSRB {split_name} samples found in: {self.data_dir}")

    def _load_train_samples(self):
        train_dir = self.data_dir / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"Missing GTSRB train folder: {train_dir}")

        samples = []
        for class_dir in sorted(path for path in train_dir.iterdir() if path.is_dir()):
            if not class_dir.name.isdigit():
                continue
            class_id = int(class_dir.name)
            for image_path in sorted(class_dir.glob("*.ppm")):
                samples.append((image_path, class_id))
        return samples

    def _load_test_samples(self):
        labels_path = self.data_dir / "test" / "labels.csv"
        images_dir = self.data_dir / "test" / "images"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing GTSRB test labels file: {labels_path}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Missing GTSRB test images folder: {images_dir}")

        samples = []
        with labels_path.open(newline="") as file:
            reader = csv.DictReader(file, delimiter=";")
            for row in reader:
                image_path = images_dir / row["Filename"]
                if image_path.exists():
                    samples.append((image_path, int(row["ClassId"])))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


def build_transforms(config, train):
    dataset_name = dataset_name_from_config(config)
    spec = dataset_spec(dataset_name)
    input_size = int(config.get("input_size", spec["input_shape"][1]))
    normalization = normalization_for_dataset(dataset_name, config)

    native_size = int(spec["input_shape"][1])

    steps = []
    if spec["kind"] == "gtsrb_folder" or input_size != native_size:
        steps.append(transforms.Resize((input_size, input_size)))
    if train and spec.get("train_augmentation") == "cifar_32":
        crop_padding = max(4, input_size // 8)
        steps.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(input_size, padding=crop_padding),
        ])
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(normalization["mean"], normalization["std"]),
    ])
    return transforms.Compose(steps)


def build_dataset(config, train=True, transform=None):
    dataset_name = dataset_name_from_config(config)
    data_dir = data_dir_from_config(config)
    spec = dataset_spec(dataset_name)
    if spec["kind"] == "cifar_batch":
        return CifarBatchDataset(data_dir, dataset_name, train=train, transform=transform)
    if spec["kind"] == "gtsrb_folder":
        return GTSRBDataset(data_dir, train=train, transform=transform)
    raise ValueError(f"Unsupported dataset kind: {spec['kind']}")



def _count_cifar_split(data_dir, spec, train):
    batch_names = spec["train_files"] if train else spec["test_files"]
    total = 0
    labels = set()
    for batch_name in batch_names:
        batch_path = Path(data_dir) / batch_name
        if not batch_path.exists():
            return None, set(), f"missing {batch_path}"
        batch = load_pickle(batch_path)
        split_labels = batch[spec["label_field"]]
        total += len(split_labels)
        labels.update(int(label) for label in split_labels)
    return total, labels, None


def _ppm_size(path):
    # PPM headers are small ASCII tokens: magic, width, height, maxval.
    with Path(path).open("rb") as file:
        tokens = []
        current = bytearray()
        in_comment = False
        while len(tokens) < 3:
            char = file.read(1)
            if not char:
                break
            if in_comment:
                if char == b"\n":
                    in_comment = False
                continue
            if char == b"#":
                in_comment = True
                continue
            if char.isspace():
                if current:
                    tokens.append(current.decode("ascii"))
                    current.clear()
            else:
                current.extend(char)
        if current and len(tokens) < 3:
            tokens.append(current.decode("ascii"))
    if len(tokens) < 3 or tokens[0] not in {"P3", "P6"}:
        return None
    return int(tokens[1]), int(tokens[2])


def _size_range(image_paths, max_images=500):
    paths = list(image_paths)
    if len(paths) > max_images:
        step = max(1, len(paths) // max_images)
        paths = paths[::step][:max_images]
    widths = []
    heights = []
    for image_path in paths:
        size = _ppm_size(image_path)
        if size is None:
            continue
        width, height = size
        widths.append(width)
        heights.append(height)
    if not widths:
        return "-"
    suffix = "" if len(paths) == len(image_paths) else f" sampled {len(paths)}"
    if min(widths) == max(widths) and min(heights) == max(heights):
        return f"{min(widths)}x{min(heights)}{suffix}"
    return f"w {min(widths)}..{max(widths)}, h {min(heights)}..{max(heights)}{suffix}"


def _count_gtsrb(data_dir):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_labels_path = data_dir / "test" / "labels.csv"
    test_images_dir = data_dir / "test" / "images"

    if not train_dir.exists():
        return None, None, set(), set(), "-", "-", f"missing {train_dir}"
    if not test_labels_path.exists() or not test_images_dir.exists():
        return None, None, set(), set(), "-", "-", f"missing GTSRB test labels/images in {data_dir / 'test'}"

    train_images = []
    train_labels = set()
    for class_dir in sorted(path for path in train_dir.iterdir() if path.is_dir() and path.name.isdigit()):
        class_id = int(class_dir.name)
        images = sorted(class_dir.glob("*.ppm"))
        if images:
            train_labels.add(class_id)
            train_images.extend(images)

    test_images = []
    test_labels = set()
    with test_labels_path.open(newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            image_path = test_images_dir / row["Filename"]
            if image_path.exists():
                test_images.append(image_path)
                test_labels.add(int(row["ClassId"]))

    return (
        len(train_images),
        len(test_images),
        train_labels,
        test_labels,
        _size_range(train_images),
        _size_range(test_images),
        None,
    )


def dataset_local_summary(dataset, data_root=Path("data")):
    dataset_name = normalize_dataset_name(dataset)
    spec = dataset_spec(dataset_name)
    data_dir = Path(data_root) / spec["folder"]
    summary = {
        "dataset": dataset_name,
        "display_name": spec["display_name"],
        "data_dir": data_dir,
        "kind": spec["kind"],
        "configured_input_shape": input_shape_for_dataset(dataset_name),
        "configured_num_classes": int(spec["num_classes"]),
        "normalization": normalization_for_dataset(dataset_name),
        "train_samples": None,
        "test_samples": None,
        "train_classes": None,
        "test_classes": None,
        "native_train_dimensions": "-",
        "native_test_dimensions": "-",
        "status": "ok",
    }

    if not data_dir.exists():
        summary["status"] = f"missing folder: {data_dir}"
        return summary

    if spec["kind"] == "cifar_batch":
        train_count, train_labels, train_error = _count_cifar_split(data_dir, spec, train=True)
        test_count, test_labels, test_error = _count_cifar_split(data_dir, spec, train=False)
        summary.update({
            "train_samples": train_count,
            "test_samples": test_count,
            "train_classes": len(train_labels),
            "test_classes": len(test_labels),
            "native_train_dimensions": "32x32",
            "native_test_dimensions": "32x32",
        })
        errors = [error for error in (train_error, test_error) if error]
        if errors:
            summary["status"] = "; ".join(errors)
        return summary

    if spec["kind"] == "gtsrb_folder":
        train_count, test_count, train_labels, test_labels, train_dims, test_dims, error = _count_gtsrb(data_dir)
        summary.update({
            "train_samples": train_count,
            "test_samples": test_count,
            "train_classes": len(train_labels),
            "test_classes": len(test_labels),
            "native_train_dimensions": train_dims,
            "native_test_dimensions": test_dims,
        })
        if error:
            summary["status"] = error
        return summary

    summary["status"] = f"unsupported kind: {spec['kind']}"
    return summary


def _format_value(value):
    if value is None:
        return "-"
    if isinstance(value, list):
        return "x".join(str(item) for item in value)
    return str(value)


def print_datasets_summary(data_root=Path("data"), datasets=None):
    datasets = datasets or sorted(DATASET_SPECS)
    rows = [dataset_local_summary(dataset, data_root=data_root) for dataset in datasets]
    dim_width = 40
    header = (
        f"{'dataset':10} | {'data_dir':18} | {'train':>7} | {'test':>7} | "
        f"{'classes':>9} | {'input':>8} | {'native train dims':{dim_width}} | "
        f"{'native test dims':{dim_width}} | status"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        classes = f"{_format_value(row['train_classes'])}/{_format_value(row['test_classes'])}"
        print(
            f"{row['dataset'][:10]:10} | "
            f"{Path(row['data_dir']).as_posix()[:18]:18} | "
            f"{_format_value(row['train_samples']):>7} | "
            f"{_format_value(row['test_samples']):>7} | "
            f"{classes:>9} | "
            f"{_format_value(row['configured_input_shape']):>8} | "
            f"{row['native_train_dimensions']:{dim_width}} | "
            f"{row['native_test_dimensions']:{dim_width}} | "
            f"{row['status']}"
        )

    print("\nNormalization:")
    for row in rows:
        normalization = row["normalization"]
        print(f"- {row['dataset']}: mean={normalization['mean']} std={normalization['std']}")


def _parse_summary_args():
    import argparse

    parser = argparse.ArgumentParser(description="Show local dataset registry and dataset sizes.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root folder containing dataset folders.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_SPECS),
        action="append",
        help="Dataset to display. Can be provided multiple times. Defaults to all datasets.",
    )
    return parser.parse_args()


def main():
    args = _parse_summary_args()
    print_datasets_summary(data_root=args.data_root, datasets=args.dataset)


if __name__ == "__main__":
    main()
