import argparse
import csv
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_DIR / "data"
DATASETS = {
    "cifar10": {
        "display_name": "CIFAR-10",
        "folder": "CIFAR-10",
        "kind": "cifar",
        "meta_file": "batches.meta",
        "label_key": "label_names",
        "label_field": "labels",
        "train_files": [f"data_batch_{index}" for index in range(1, 6)],
        "test_files": ["test_batch"],
    },
    "cifar100": {
        "display_name": "CIFAR-100",
        "folder": "CIFAR-100",
        "kind": "cifar",
        "meta_file": "meta",
        "label_key": "fine_label_names",
        "label_field": "fine_labels",
        "train_files": ["train"],
        "test_files": ["test"],
    },
    "gtsrb": {
        "display_name": "GTSRB",
        "folder": "GTSRB",
        "kind": "gtsrb",
        "num_classes": 43,
    },
}
DATASET_ALIASES = {"gtrsb": "gtsrb"}
DATASET_CHOICES = sorted([*DATASETS.keys(), *DATASET_ALIASES.keys()])


def normalize_dataset_name(dataset):
    return DATASET_ALIASES.get(dataset.lower(), dataset.lower())


def load_pickle(path):
    with path.open("rb") as file:
        return pickle.load(file, encoding="latin1")


def dataset_dir(data_root, dataset):
    dataset = normalize_dataset_name(dataset)
    return data_root / DATASETS[dataset]["folder"]


def available_datasets(data_root):
    return [name for name in DATASETS if dataset_dir(data_root, name).exists()]


def load_label_names(data_dir, dataset):
    dataset = normalize_dataset_name(dataset)
    config = DATASETS[dataset]
    if config["kind"] == "gtsrb":
        return [f"class_{class_id:05d}" for class_id in range(config["num_classes"])]

    metadata = load_pickle(data_dir / config["meta_file"])
    return metadata[config["label_key"]]


def load_cifar_batch(path, dataset):
    config = DATASETS[dataset]
    batch = load_pickle(path)
    images = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(batch[config["label_field"]])
    filenames = batch.get("filenames", [""] * len(labels))
    return images, labels, filenames


def cifar_batch_paths(data_dir, dataset, split):
    config = DATASETS[dataset]
    if split == "train":
        filenames = config["train_files"]
    elif split == "test":
        filenames = config["test_files"]
    else:
        filenames = config["train_files"] + config["test_files"]

    return [data_dir / filename for filename in filenames]


def load_cifar_split(data_dir, dataset, split):
    images_list = []
    labels_list = []
    filenames_list = []

    for path in cifar_batch_paths(data_dir, dataset, split):
        if not path.exists():
            raise SystemExit(f"Missing CIFAR batch file: {path}")
        images, labels, filenames = load_cifar_batch(path, dataset)
        images_list.append(images)
        labels_list.append(labels)
        filenames_list.extend(filenames)

    return np.concatenate(images_list), np.concatenate(labels_list), filenames_list


def load_gtsrb_train_split(data_dir):
    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise SystemExit(f"Missing GTSRB train folder: {train_dir}")

    image_paths = []
    labels = []
    filenames = []

    for class_dir in sorted(path for path in train_dir.iterdir() if path.is_dir()):
        if not class_dir.name.isdigit():
            continue
        class_id = int(class_dir.name)
        for image_path in sorted(class_dir.glob("*.ppm")):
            image_paths.append(image_path)
            labels.append(class_id)
            filenames.append(str(image_path.relative_to(data_dir)))

    return image_paths, np.array(labels), filenames


def load_gtsrb_test_split(data_dir):
    labels_path = data_dir / "test" / "labels.csv"
    images_dir = data_dir / "test" / "images"
    if not labels_path.exists():
        raise SystemExit(f"Missing GTSRB test labels file: {labels_path}")
    if not images_dir.exists():
        raise SystemExit(f"Missing GTSRB test images folder: {images_dir}")

    image_paths = []
    labels = []
    filenames = []

    with labels_path.open(newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            image_path = images_dir / row["Filename"]
            if not image_path.exists():
                continue
            image_paths.append(image_path)
            labels.append(int(row["ClassId"]))
            filenames.append(str(image_path.relative_to(data_dir)))

    return image_paths, np.array(labels), filenames


def load_gtsrb_split(data_dir, split):
    if split == "train":
        return load_gtsrb_train_split(data_dir)
    if split == "test":
        return load_gtsrb_test_split(data_dir)

    train_images, train_labels, train_filenames = load_gtsrb_train_split(data_dir)
    test_images, test_labels, test_filenames = load_gtsrb_test_split(data_dir)
    return train_images + test_images, np.concatenate([train_labels, test_labels]), train_filenames + test_filenames


def load_split(data_dir, dataset, split):
    dataset = normalize_dataset_name(dataset)
    config = DATASETS[dataset]
    if config["kind"] == "gtsrb":
        return load_gtsrb_split(data_dir, split)
    return load_cifar_split(data_dir, dataset, split)


def select_examples(labels, samples_per_class, seed, class_id=None):
    rng = np.random.default_rng(seed)
    selected_indices = []
    class_ids = [class_id] if class_id is not None else sorted(np.unique(labels))

    for current_class_id in class_ids:
        matches = np.flatnonzero(labels == current_class_id)
        count = min(samples_per_class, len(matches))
        if count:
            selected_indices.extend(rng.choice(matches, size=count, replace=False))

    return np.array(selected_indices)


def resolve_class_id(class_name, label_names):
    if class_name is None:
        return None

    value = class_name.strip().lower()
    if value.isdigit():
        class_id = int(value)
        if 0 <= class_id < len(label_names):
            return class_id

    normalized_names = {name.lower(): index for index, name in enumerate(label_names)}
    class_id = normalized_names.get(value)
    if class_id is None:
        available = ", ".join(label_names)
        raise SystemExit(f"Unknown class '{class_name}'. Available classes: {available}")
    return class_id


def load_image_for_display(image):
    if isinstance(image, Path):
        with Image.open(image) as opened_image:
            return np.asarray(opened_image.convert("RGB"))
    return image


def plot_examples(images, labels, label_names, indices, columns):
    if len(indices) == 0:
        raise SystemExit("No image found for the requested selection.")

    cols = max(1, columns)
    rows = max(1, int(np.ceil(len(indices) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.1))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for axis in axes.ravel():
        axis.axis("off")

    for axis, index in zip(axes.ravel(), indices):
        label = labels[index]
        axis.imshow(load_image_for_display(images[index]))
        axis.set_title(label_names[label], fontsize=8)
        axis.axis("off")

    fig.tight_layout()
    plt.show()


def ask_choice(title, options, default_index=0):
    print(f"\n{title}")
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")

    while True:
        answer = input(f"Choix [{default_index + 1}]: ").strip()
        if not answer:
            return default_index
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print("Choix invalide.")


def ask_int(prompt, default, minimum=1, maximum=None, default_label=None):
    shown_default = f"{default_label} : {default}" if default_label else str(default)
    while True:
        answer = input(f"{prompt} [{shown_default}]: ").strip()
        if not answer:
            return default
        if answer.isdigit():
            value = int(answer)
            if value >= minimum and (maximum is None or value <= maximum):
                return value
        limit = f" entre {minimum} et {maximum}" if maximum is not None else f" >= {minimum}"
        print(f"Entrez un entier{limit}.")


def ask_optional_class(label_names):
    mode = ask_choice("Que veux-tu afficher ?", ["Toutes les classes", "Une classe precise"])
    if mode == 0:
        return None

    print("\nClasses disponibles:")
    for index, name in enumerate(label_names):
        print(f"  {index}: {name}")

    while True:
        answer = input("Nom ou numero de classe: ").strip()
        try:
            return resolve_class_id(answer, label_names)
        except SystemExit as error:
            print(error)


def interactive_args():
    data_root = DEFAULT_DATA_ROOT
    datasets = available_datasets(data_root)
    if not datasets:
        raise SystemExit(f"No supported dataset folder found in: {data_root}")

    dataset_options = [DATASETS[name]["display_name"] for name in datasets]
    dataset = datasets[ask_choice("Datasets disponibles", dataset_options)]
    data_dir = dataset_dir(data_root, dataset)
    label_names = load_label_names(data_dir, dataset)

    split = ["train", "test", "all"][ask_choice("Split a visualiser", ["train", "test", "all"])]
    class_id = ask_optional_class(label_names)
    samples_per_class = ask_int("Images par classe", 8, minimum=1, maximum=100)

    max_classes = None
    if class_id is None:
        max_classes = ask_int("Nombre maximum de classes", len(label_names), minimum=1, maximum=len(label_names), default_label="max")

    default_columns = samples_per_class if class_id is None else min(samples_per_class, 8)
    columns = ask_int("Nombre de colonnes", default_columns, minimum=1, maximum=20)
    seed = ask_int("Seed aleatoire", 7, minimum=0)

    return argparse.Namespace(
        data_root=data_root,
        dataset=dataset,
        split=split,
        samples_per_class=samples_per_class,
        class_id=class_id,
        max_classes=max_classes,
        columns=columns,
        seed=seed,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize CIFAR-10, CIFAR-100, or GTSRB samples from the local data folder.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Folder containing CIFAR-10, CIFAR-100, and/or GTSRB folders.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="cifar10", help="Dataset to visualize. The typo alias 'gtrsb' is accepted for GTSRB.")
    parser.add_argument("--split", choices=["train", "test", "all"], default="train", help="Data split to visualize.")
    parser.add_argument("--samples-per-class", type=int, default=8, help="Number of examples per class.")
    parser.add_argument("--class-name", help="Optional class name or class id to visualize, for example: cat, dog, truck, apple, 14, class_00014.")
    parser.add_argument("--max-classes", type=int, help="Limit the number of classes shown, useful for CIFAR-100 and GTSRB.")
    parser.add_argument("--columns", type=int, help="Number of image columns in the output grid.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used to pick samples.")
    return parser.parse_args()


def build_args():
    if len(sys.argv) == 1:
        return interactive_args()

    args = parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    data_dir = dataset_dir(args.data_root, args.dataset)
    label_names = load_label_names(data_dir, args.dataset) if data_dir.exists() else []
    args.class_id = resolve_class_id(args.class_name, label_names) if args.class_name else None
    return args


def main():
    args = build_args()
    data_dir = dataset_dir(args.data_root, args.dataset)
    if not data_dir.exists():
        raise SystemExit(f"Dataset folder not found: {data_dir}")

    label_names = load_label_names(data_dir, args.dataset)
    images, labels, _ = load_split(data_dir, args.dataset, args.split)
    indices = select_examples(labels, args.samples_per_class, args.seed, args.class_id)

    if args.max_classes and args.class_id is None:
        max_images = args.max_classes * args.samples_per_class
        indices = indices[:max_images]

    columns = args.columns or args.samples_per_class
    plot_examples(images, labels, label_names, indices, columns)


if __name__ == "__main__":
    main()
