import csv
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DATASET_ALIASES = {
    "gtrsb": "gtsrb",
    "sturm": "sturm_flood",
    "sturm-flood": "sturm_flood",
    "sen1": "sen1floods11",
    "sen1floods": "sen1floods11",
    "sen1-floods11": "sen1floods11",
}

SEN1FLOODS11_SOURCES = {
    "hand_s1": {
        "display_name": "Hand labeled S1",
        "label": 0,
        "source_folder": ("flood_events", "HandLabeled", "S1Hand"),
        "source_suffix": "_S1Hand.tif",
        "mask_folder": ("flood_events", "HandLabeled", "LabelHand"),
        "mask_suffix": "_LabelHand.tif",
    },
    "hand_s2": {
        "display_name": "Hand labeled S2",
        "label": 1,
        "source_folder": ("flood_events", "HandLabeled", "S2Hand"),
        "source_suffix": "_S2Hand.tif",
        "mask_folder": ("flood_events", "HandLabeled", "LabelHand"),
        "mask_suffix": "_LabelHand.tif",
    },
    "weak_s1": {
        "display_name": "Weak labeled S1",
        "label": 2,
        "source_folder": ("flood_events", "WeaklyLabeled", "S1Weak"),
        "source_suffix": "_S1Weak.tif",
        "mask_folder": ("flood_events", "WeaklyLabeled", "S1OtsuLabelWeak"),
        "mask_suffix": "_S1OtsuLabelWeak.tif",
    },
    "weak_s2": {
        "display_name": "Weak labeled S2",
        "label": 3,
        "source_folder": ("flood_events", "WeaklyLabeled", "S2Weak"),
        "source_suffix": "_S2Weak.tif",
        "mask_folder": ("flood_events", "WeaklyLabeled", "S2IndexLabelWeak"),
        "mask_suffix": "_S2IndexLabelWeak.tif",
    },
    "perm_water_s1": {
        "display_name": "Permanent water S1",
        "label": 4,
        "source_folder": ("perm_water", "S1Perm"),
        "source_prefix": "sentinel_",
        "mask_folder": ("perm_water", "JRCPerm"),
        "mask_prefix": "water_",
    },
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
    "sturm_flood": {
        "display_name": "STURM-Flood",
        "folder": "STURM-Flood",
        "kind": "sturm_flood",
        "num_classes": 2,
        "input_shape": "GeoTIFF 128x128",
        "normalization": {
            "mean": [],
            "std": [],
        },
        "sources": ["sentinel1", "sentinel2"],
    },
    "sen1floods11": {
        "display_name": "Sen1Floods11",
        "folder": "Sen1Floods11",
        "kind": "sen1floods11",
        "num_classes": len(SEN1FLOODS11_SOURCES),
        "input_shape": "GeoTIFF 512x512",
        "normalization": {
            "mean": [],
            "std": [],
        },
        "sources": list(SEN1FLOODS11_SOURCES),
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
    if spec["kind"] in {"sturm_flood", "sen1floods11"}:
        return spec["input_shape"]
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
    if spec["kind"] == "sturm_flood":
        return ["sentinel1", "sentinel2"]
    if spec["kind"] == "sen1floods11":
        return list(SEN1FLOODS11_SOURCES)
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
    if spec["kind"] == "sturm_flood":
        raise ValueError("STURM-Flood is available for visualization only; training loaders are not implemented yet.")
    if spec["kind"] == "sen1floods11":
        raise ValueError("Sen1Floods11 is available for visualization only; training loaders are not implemented yet.")
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


def _sen1_mask_name(source_name, config):
    if "source_suffix" in config:
        return source_name.replace(config["source_suffix"], config["mask_suffix"])
    if "source_prefix" in config:
        return source_name.replace(config["source_prefix"], config["mask_prefix"], 1)
    return source_name


def _count_sen1floods11(data_dir):
    data_path = Path(data_dir) / "dataset" / "v1.1" / "data"
    required = []
    source_counts = []
    labels = set()
    status_messages = []

    for source_name, config in SEN1FLOODS11_SOURCES.items():
        source_dir = data_path.joinpath(*config["source_folder"])
        mask_dir = data_path.joinpath(*config["mask_folder"])
        if not source_dir.exists():
            required.append(source_dir)
            continue
        if not mask_dir.exists():
            required.append(mask_dir)
            continue

        source_files = sorted(source_dir.glob("*.tif"))
        paired_count = 0
        missing_masks = 0
        for source_path in source_files:
            mask_path = mask_dir / _sen1_mask_name(source_path.name, config)
            if mask_path.exists():
                paired_count += 1
            else:
                missing_masks += 1

        source_counts.append(paired_count)
        if paired_count:
            labels.add(config["label"])
        if missing_masks:
            status_messages.append(f"{source_name}: {missing_masks} missing masks")

    if required:
        return None, set(), "-", "missing " + ", ".join(str(path) for path in required)

    status = "; ".join(status_messages) if status_messages else None
    return sum(source_counts), labels, "512x512 GeoTIFF source+mask", status


def _count_sturm_flood(data_dir):
    dataset_dir_path = Path(data_dir) / "dataset"
    sentinel1_source = dataset_dir_path / "Sentinel1" / "S1"
    sentinel1_masks = dataset_dir_path / "Sentinel1" / "Floodmaps"
    sentinel2_source = dataset_dir_path / "Sentinel2" / "S2"
    sentinel2_masks = dataset_dir_path / "Sentinel2" / "Floodmaps"

    required = [sentinel1_source, sentinel1_masks, sentinel2_source, sentinel2_masks]
    missing = [path for path in required if not path.exists()]
    if missing:
        return None, None, set(), set(), "-", "-", "missing " + ", ".join(str(path) for path in missing)

    sentinel1_files = sorted(sentinel1_source.glob("*.tif"))
    sentinel2_files = sorted(sentinel2_source.glob("*.tif"))
    sentinel1_mask_count = len(list(sentinel1_masks.glob("*.tif")))
    sentinel2_mask_count = len(list(sentinel2_masks.glob("*.tif")))

    status = None
    if len(sentinel1_files) != sentinel1_mask_count or len(sentinel2_files) != sentinel2_mask_count:
        status = (
            f"source/mask mismatch: sentinel1 {len(sentinel1_files)}/{sentinel1_mask_count}, "
            f"sentinel2 {len(sentinel2_files)}/{sentinel2_mask_count}"
        )

    return (
        len(sentinel1_files),
        len(sentinel2_files),
        {0} if sentinel1_files else set(),
        {1} if sentinel2_files else set(),
        "128x128 GeoTIFF source+mask",
        "128x128 GeoTIFF source+mask",
        status,
    )


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
        "total_samples": None,
        "status": "ok",
    }

    if not data_dir.exists():
        summary["status"] = f"missing folder: {data_dir}"
        return summary

    if spec["kind"] == "cifar_batch":
        train_count, train_labels, train_error = _count_cifar_split(data_dir, spec, train=True)
        test_count, test_labels, test_error = _count_cifar_split(data_dir, spec, train=False)
        summary.update({
            "total_samples": None if train_count is None or test_count is None else train_count + test_count,
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
            "total_samples": None if train_count is None or test_count is None else train_count + test_count,
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

    if spec["kind"] == "sturm_flood":
        train_count, test_count, train_labels, test_labels, train_dims, test_dims, error = _count_sturm_flood(data_dir)
        total_count = None if train_count is None or test_count is None else train_count + test_count
        summary.update({
            "total_samples": total_count,
            "train_samples": "/",
            "test_samples": "/",
            "train_classes": len(train_labels | test_labels),
            "test_classes": "/",
            "native_train_dimensions": train_dims,
            "native_test_dimensions": test_dims,
        })
        if error:
            summary["status"] = error
        return summary

    if spec["kind"] == "sen1floods11":
        total_count, labels, dims, error = _count_sen1floods11(data_dir)
        summary.update({
            "total_samples": total_count,
            "train_samples": "/",
            "test_samples": "/",
            "train_classes": len(labels),
            "test_classes": "/",
            "native_train_dimensions": dims,
            "native_test_dimensions": "/",
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
    columns = [
        ("dataset", "dataset", "left"),
        ("data_dir", "data_dir", "left"),
        ("total", "total_samples", "right"),
        ("train", "train_samples", "right"),
        ("test", "test_samples", "right"),
        ("classes", "classes", "right"),
        ("input", "configured_input_shape", "left"),
        ("train dims", "native_train_dimensions", "left"),
        ("test dims", "native_test_dimensions", "left"),
        ("status", "status", "left"),
    ]

    display_rows = []
    for row in rows:
        display_row = dict(row)
        display_row["data_dir"] = Path(row["data_dir"]).as_posix()
        if row["test_classes"] == "/":
            display_row["classes"] = _format_value(row["train_classes"])
        else:
            display_row["classes"] = f"{_format_value(row['train_classes'])}/{_format_value(row['test_classes'])}"
        display_row["configured_input_shape"] = _format_value(row["configured_input_shape"])
        display_row["total_samples"] = _format_value(row["total_samples"])
        display_row["train_samples"] = _format_value(row["train_samples"])
        display_row["test_samples"] = _format_value(row["test_samples"])
        display_rows.append(display_row)

    widths = {}
    for title, key, _ in columns:
        widths[key] = max(len(title), *(len(str(row[key])) for row in display_rows))

    def cell(value, key, align):
        value = str(value)
        if align == "right":
            return value.rjust(widths[key])
        return value.ljust(widths[key])

    header = " | ".join(cell(title, key, align) for title, key, align in columns)
    print(header)
    print("-" * len(header))
    for row in display_rows:
        print(" | ".join(cell(row[key], key, align) for _, key, align in columns))

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
