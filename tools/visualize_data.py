import argparse
import csv
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools.datasets import (
    DATASET_ALIASES,
    DATASET_SPECS,
    FLOODCASTBENCH_SOURCES,
    SEN1FLOODS11_SOURCES,
    STURM_FLOOD_SOURCES,
    class_names_for_dataset,
    normalize_dataset_name,
)

DEFAULT_DATA_ROOT = PROJECT_DIR / "data"


def visualizer_dataset_config(name, spec):
    kind = "cifar"
    if spec["kind"] == "gtsrb_folder":
        kind = "gtsrb"
    elif spec["kind"] == "sturm_flood":
        kind = "sturm_flood"
    elif spec["kind"] == "sen1floods11":
        kind = "sen1floods11"
    elif spec["kind"] == "floodcastbench":
        kind = "floodcastbench"
    config = {
        "display_name": spec["display_name"],
        "folder": spec["folder"],
        "kind": kind,
        "num_classes": spec["num_classes"],
    }
    if spec["kind"] == "cifar_batch":
        config.update({
            "meta_file": spec["meta_file"],
            "label_key": spec["label_names_key"],
            "label_field": spec["label_field"],
            "train_files": spec["train_files"],
            "test_files": spec["test_files"],
        })
    return config


DATASETS = {name: visualizer_dataset_config(name, spec) for name, spec in DATASET_SPECS.items()}
DATASET_CHOICES = sorted({*DATASETS.keys(), *DATASET_ALIASES.keys()})

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
    if config["kind"] in {"sturm_flood", "sen1floods11", "floodcastbench", "gtsrb"}:
        return class_names_for_dataset(dataset, data_dir)

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


def sturm_dataset_dir(data_dir):
    dataset_dir_path = data_dir / "dataset"
    if not dataset_dir_path.exists():
        raise SystemExit(f"Missing STURM-Flood dataset folder: {dataset_dir_path}")
    return dataset_dir_path


def sturm_source_key(source):
    aliases = {"Sentinel1": "sentinel1", "Sentinel2": "sentinel2", "train": "sentinel1", "test": "sentinel2"}
    return aliases.get(source, source)


def load_sturm_sensor_split(data_dir, source, label=None):
    source_key = sturm_source_key(source)
    if source_key not in STURM_FLOOD_SOURCES:
        available = ", ".join([*STURM_FLOOD_SOURCES, "all"])
        raise SystemExit(f"For STURM-Flood, --split must be one of: {available}.")

    config = STURM_FLOOD_SOURCES[source_key]
    root = sturm_dataset_dir(data_dir) / config["sensor_folder"]
    source_dir = root / config["source_folder"]
    mask_dir = root / config["mask_folder"]
    label = config["label"] if label is None else label

    if not source_dir.exists():
        raise SystemExit(f"Missing STURM-Flood source folder: {source_dir}")
    if not mask_dir.exists():
        raise SystemExit(f"Missing STURM-Flood floodmap folder: {mask_dir}")

    samples = []
    labels = []
    filenames = []
    for source_path in sorted(source_dir.glob("*.tif")):
        mask_path = mask_dir / source_path.name
        if not mask_path.exists():
            continue
        samples.append((source_path, mask_path, source_key))
        labels.append(label)
        filenames.append(str(source_path.relative_to(data_dir)))
    return samples, np.array(labels), filenames


def sturm_source_options(split):
    if split == "all":
        return list(STURM_FLOOD_SOURCES)
    source_key = sturm_source_key(split)
    if source_key not in STURM_FLOOD_SOURCES:
        available = ", ".join([*STURM_FLOOD_SOURCES, "all"])
        raise SystemExit(f"For STURM-Flood, --split must be one of: {available}.")
    return [source_key]


def load_sturm_split(data_dir, split):
    all_images = []
    all_labels = []
    all_filenames = []
    for source_key in sturm_source_options(split):
        images, labels, filenames = load_sturm_sensor_split(data_dir, source_key)
        all_images.extend(images)
        all_labels.append(labels)
        all_filenames.extend(filenames)
    if not all_labels:
        return [], np.array([], dtype=int), []
    return all_images, np.concatenate(all_labels), all_filenames


def sen1_dataset_dir(data_dir):
    data_path = data_dir / "dataset" / "v1.1" / "data"
    if not data_path.exists():
        raise SystemExit(f"Missing Sen1Floods11 data folder: {data_path}")
    return data_path


def sen1_source_options(split):
    if split == "all":
        return list(SEN1FLOODS11_SOURCES)
    if split not in SEN1FLOODS11_SOURCES:
        available = ", ".join([*SEN1FLOODS11_SOURCES, "all"])
        raise SystemExit(f"For Sen1Floods11, --split must be one of: {available}.")
    return [split]


def sen1_mask_name(source_name, config):
    if "source_suffix" in config:
        return source_name.replace(config["source_suffix"], config["mask_suffix"])
    if "source_prefix" in config:
        return source_name.replace(config["source_prefix"], config["mask_prefix"], 1)
    return source_name


def load_sen1_source(data_dir, source_name):
    data_path = sen1_dataset_dir(data_dir)
    config = SEN1FLOODS11_SOURCES[source_name]
    source_dir = data_path.joinpath(*config["source_folder"])
    mask_dir = data_path.joinpath(*config["mask_folder"])
    if not source_dir.exists():
        raise SystemExit(f"Missing Sen1Floods11 source folder: {source_dir}")
    if not mask_dir.exists():
        raise SystemExit(f"Missing Sen1Floods11 mask folder: {mask_dir}")

    samples = []
    labels = []
    filenames = []
    for source_path in sorted(source_dir.glob("*.tif")):
        mask_path = mask_dir / sen1_mask_name(source_path.name, config)
        if not mask_path.exists():
            continue
        samples.append((source_path, mask_path, source_name))
        labels.append(config["label"])
        filenames.append(str(source_path.relative_to(data_dir)))
    return samples, np.array(labels), filenames


def load_sen1_split(data_dir, split):
    all_images = []
    all_labels = []
    all_filenames = []
    for source_name in sen1_source_options(split):
        images, labels, filenames = load_sen1_source(data_dir, source_name)
        all_images.extend(images)
        all_labels.append(labels)
        all_filenames.extend(filenames)
    if not all_labels:
        return [], np.array([], dtype=int), []
    return all_images, np.concatenate(all_labels), all_filenames


def floodcast_source_options(split):
    if split == "all":
        return list(FLOODCASTBENCH_SOURCES)
    if split not in FLOODCASTBENCH_SOURCES:
        available = ", ".join([*FLOODCASTBENCH_SOURCES, "all"])
        raise SystemExit(f"For FloodCastBench, --split must be one of: {available}.")
    return [split]


def floodcast_sort_key(path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem.lower())


def load_floodcast_source(data_dir, source_name):
    config = FLOODCASTBENCH_SOURCES[source_name]
    source_dir = data_dir.joinpath(*config["folder"])
    if not source_dir.exists():
        raise SystemExit(f"Missing FloodCastBench folder: {source_dir}")

    image_paths = sorted(
        [*source_dir.rglob("*.tif"), *source_dir.rglob("*.tiff")],
        key=floodcast_sort_key,
    )
    labels = np.full(len(image_paths), config["label"], dtype=int)
    filenames = [str(image_path.relative_to(data_dir)) for image_path in image_paths]
    return image_paths, labels, filenames


def load_floodcast_split(data_dir, split):
    all_images = []
    all_labels = []
    all_filenames = []
    for source_name in floodcast_source_options(split):
        images, labels, filenames = load_floodcast_source(data_dir, source_name)
        all_images.extend(images)
        all_labels.append(labels)
        all_filenames.extend(filenames)
    if not all_labels:
        return [], np.array([], dtype=int), []
    return all_images, np.concatenate(all_labels), all_filenames


def load_split(data_dir, dataset, split):
    dataset = normalize_dataset_name(dataset)
    config = DATASETS[dataset]
    if config["kind"] == "sturm_flood":
        return load_sturm_split(data_dir, split)
    if config["kind"] == "sen1floods11":
        return load_sen1_split(data_dir, split)
    if config["kind"] == "floodcastbench":
        return load_floodcast_split(data_dir, split)
    if split not in {"train", "test", "all"}:
        raise SystemExit(f"For {dataset}, --split must be train, test, or all.")
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


def temporal_indices(matches, count, mode, rng):
    if count <= 0:
        return []
    if mode == "first":
        return matches[:count]
    if mode == "middle":
        start = max(0, (len(matches) - count) // 2)
        return matches[start:start + count]
    if mode == "last":
        return matches[-count:]
    if mode == "evenly_spaced":
        positions = np.linspace(0, len(matches) - 1, num=count, dtype=int)
        return matches[positions]
    return rng.choice(matches, size=count, replace=False)


def select_floodcast_examples(labels, samples_per_class, seed, mode):
    rng = np.random.default_rng(seed)
    selected_indices = []
    for current_label in sorted(np.unique(labels)):
        matches = np.flatnonzero(labels == current_label)
        count = min(samples_per_class, len(matches))
        selected_indices.extend(temporal_indices(matches, count, mode, rng))
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
    if isinstance(image, tuple):
        return load_geotiff_pair_for_display(image)
    if isinstance(image, Path):
        if image.suffix.lower() in {".tif", ".tiff"}:
            return geotiff_source_to_rgb(image)
        with Image.open(image) as opened_image:
            return np.asarray(opened_image.convert("RGB"))
    return image


def read_geotiff(path):
    try:
        import rasterio
    except ImportError as error:
        raise SystemExit(
            "GeoTIFF datasets use rasterio. Install it with: "
            "python -m pip install -r requirements.txt"
        ) from error

    with rasterio.open(path) as source:
        return source.read()


def stretch_to_uint8(array):
    array = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(array)
    if not valid.any():
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.percentile(array[valid], [2, 98])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def geotiff_source_to_rgb(path):
    data = read_geotiff(path)
    if data.ndim != 3:
        raise SystemExit(f"Unexpected GeoTIFF shape for {path}: {data.shape}")

    if data.shape[0] >= 4:
        rgb = np.stack([data[3], data[2], data[1]], axis=-1)
    elif data.shape[0] >= 3:
        rgb = np.moveaxis(data[:3], 0, -1)
    elif data.shape[0] == 2:
        mean_band = (data[0] + data[1]) / 2.0
        rgb = np.stack([data[0], data[1], mean_band], axis=-1)
    else:
        rgb = np.repeat(data[0][..., None], 3, axis=-1)
    return stretch_to_uint8(rgb)


def geotiff_mask_to_rgb(path):
    data = read_geotiff(path)[0]
    mask = data > 0
    rgb = np.full((*data.shape, 3), 245, dtype=np.uint8)
    rgb[mask] = np.array([30, 120, 220], dtype=np.uint8)
    return rgb


def load_geotiff_pair_for_display(sample):
    source_path, mask_path, _ = sample
    source_rgb = geotiff_source_to_rgb(source_path)
    mask_rgb = geotiff_mask_to_rgb(mask_path)
    separator = np.full((source_rgb.shape[0], 4, 3), 255, dtype=np.uint8)
    return np.concatenate([source_rgb, separator, mask_rgb], axis=1)


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


def is_floodcast_temporal_source(split):
    if split not in FLOODCASTBENCH_SOURCES:
        return False
    return FLOODCASTBENCH_SOURCES[split]["category"] in {"forecast", "rainfall"}


def is_floodcast_video_source(split):
    return is_floodcast_temporal_source(split)


def plot_floodcast_video(images, label_names, label_id, split, step, interval):
    if not is_floodcast_video_source(split):
        raise SystemExit("Le mode video est disponible seulement pour forecast/rainfall FloodCastBench.")

    frame_paths = list(images)[::max(1, step)]
    if not frame_paths:
        raise SystemExit("No frame found for FloodCastBench video mode.")

    fig, axis = plt.subplots(figsize=(6, 5))
    first_frame = load_image_for_display(frame_paths[0])
    image_artist = axis.imshow(first_frame)
    axis.axis("off")
    title = axis.set_title("", fontsize=9)

    def update(frame_index):
        frame_path = frame_paths[frame_index]
        image_artist.set_data(load_image_for_display(frame_path))
        title.set_text(
            f"{label_names[label_id]} | frame {frame_index + 1}/{len(frame_paths)} | {frame_path.stem}"
        )
        return image_artist, title

    from matplotlib.animation import FuncAnimation

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_paths),
        interval=max(1, interval),
        repeat=True,
        blit=False,
    )
    update(0)
    fig.tight_layout()
    # Mark as drawn to avoid a false-positive warning in non-interactive test backends.
    animation._draw_was_started = True
    plot_floodcast_video._animation = animation
    plt.show()
    return animation


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



def dataset_kind(dataset):
    return DATASETS[normalize_dataset_name(dataset)]["kind"]


def is_classification_dataset(dataset):
    return dataset_kind(dataset) in {"cifar", "gtsrb"}


def default_split_for_dataset(dataset):
    kind = dataset_kind(dataset)
    if kind in {"cifar", "gtsrb"}:
        return "train"
    if kind == "sturm_flood":
        return "sentinel1"
    if kind == "sen1floods11":
        return "hand_s1"
    if kind == "floodcastbench":
        return "low_mozambique_480m"
    return "all"


def split_choices_for_dataset(dataset):
    kind = dataset_kind(dataset)
    if kind in {"cifar", "gtsrb"}:
        return ["train", "test", "all"]
    if kind == "sturm_flood":
        return [*STURM_FLOOD_SOURCES, "all"]
    if kind == "sen1floods11":
        return [*SEN1FLOODS11_SOURCES, "all"]
    if kind == "floodcastbench":
        return [*FLOODCASTBENCH_SOURCES, "all"]
    return ["all"]


def validate_split_for_dataset(dataset, split):
    choices = split_choices_for_dataset(dataset)
    if split not in choices:
        available = ", ".join(choices)
        raise SystemExit(f"For {dataset}, --split must be one of: {available}.")
    return split


def floodcast_menu():
    category_options = [
        ("low", "Low-fidelity flood forecasting"),
        ("high", "High-fidelity flood forecasting"),
        ("rainfall", "Rainfall"),
        ("dem", "DEM"),
        ("land_cover", "Land use / land cover"),
        ("initial", "Initial conditions"),
        ("all", "Toutes les donnees FloodCastBench"),
    ]
    category = category_options[ask_choice("Type de donnees FloodCastBench", [label for _, label in category_options])][0]

    if category == "all":
        return "all"
    if category == "dem":
        return "dem"
    if category == "land_cover":
        return "land_cover"

    if category == "low":
        options = [
            ("low_mozambique_480m", "Mozambique - 480m"),
            ("low_pakistan_480m", "Pakistan - 480m"),
        ]
    elif category == "high":
        options = [
            ("high_australia_30m", "Australia - 30m"),
            ("high_australia_60m", "Australia - 60m"),
            ("high_uk_30m", "UK - 30m"),
            ("high_uk_60m", "UK - 60m"),
        ]
    elif category == "rainfall":
        options = [
            ("rainfall_australia", "Australia flood"),
            ("rainfall_mozambique", "Mozambique flood"),
            ("rainfall_pakistan", "Pakistan flood"),
            ("rainfall_uk", "UK flood"),
        ]
    else:
        options = [
            ("initial_high", "High-fidelity initial conditions"),
            ("initial_low", "Low-fidelity initial conditions"),
        ]

    return options[ask_choice("Evenement / resolution", [label for _, label in options])][0]


def ask_floodcast_time_mode(split):
    if not is_floodcast_temporal_source(split):
        return "random"

    options = [
        ("evenly_spaced", "Reparties sur toute la sequence"),
        ("video", "Video interactive"),
        ("first", "Debut de sequence"),
        ("middle", "Milieu de sequence"),
        ("last", "Fin de sequence"),
        ("random", "Aleatoire"),
    ]
    return options[ask_choice("Selection temporelle", [label for _, label in options])][0]


def source_menu(dataset):
    kind = dataset_kind(dataset)
    if kind == "sturm_flood":
        keys = [*STURM_FLOOD_SOURCES, "all"]
        labels = [config["display_name"] for config in STURM_FLOOD_SOURCES.values()] + ["Toutes"]
        return "Source a visualiser", keys, labels
    if kind == "sen1floods11":
        keys = [*SEN1FLOODS11_SOURCES, "all"]
        labels = [config["display_name"] for config in SEN1FLOODS11_SOURCES.values()] + ["Toutes"]
        return "Source a visualiser", keys, labels
    if kind == "floodcastbench":
        keys = [*FLOODCASTBENCH_SOURCES, "all"]
        labels = [config["display_name"] for config in FLOODCASTBENCH_SOURCES.values()] + ["Toutes"]
        return "Bloc a visualiser", keys, labels
    return "Split a visualiser", ["train", "test", "all"], ["train", "test", "all"]

def interactive_args():
    data_root = DEFAULT_DATA_ROOT
    datasets = available_datasets(data_root)
    if not datasets:
        raise SystemExit(f"No supported dataset folder found in: {data_root}")

    dataset_options = [DATASETS[name]["display_name"] for name in datasets]
    dataset = datasets[ask_choice("Datasets disponibles", dataset_options)]
    data_dir = dataset_dir(data_root, dataset)
    label_names = load_label_names(data_dir, dataset)

    if dataset_kind(dataset) == "floodcastbench":
        split = floodcast_menu()
        time_mode = ask_floodcast_time_mode(split)
    else:
        split_title, split_keys, split_labels = source_menu(dataset)
        split = split_keys[ask_choice(split_title, split_labels)]
        time_mode = "random"

    class_id = None
    max_classes = None
    video_step = 10
    video_interval = 120

    if dataset_kind(dataset) == "floodcastbench" and time_mode == "video":
        if not is_floodcast_video_source(split):
            raise SystemExit("Le mode video est disponible seulement pour forecast/rainfall FloodCastBench.")
        samples_per_class = 1
        columns = 1
        seed = 7
        video_step = ask_int("Afficher une frame toutes les N images", 10, minimum=1, maximum=10000)
        video_interval = ask_int("Delai entre frames en ms", 120, minimum=1, maximum=10000)
    else:
        if is_classification_dataset(dataset):
            class_id = ask_optional_class(label_names)
            sample_prompt = "Images par classe"
            default_samples = 8
            default_columns = min(default_samples, 8)
        else:
            if dataset_kind(dataset) == "floodcastbench":
                sample_prompt = "Images par bloc"
                default_samples = 4
                default_columns = 2
            else:
                sample_prompt = "Paires source/masque par source"
                default_samples = 4
                default_columns = 1

        samples_per_class = ask_int(sample_prompt, default_samples, minimum=1, maximum=100)

        if is_classification_dataset(dataset) and class_id is None:
            max_classes = ask_int(
                "Nombre maximum de classes",
                len(label_names),
                minimum=1,
                maximum=len(label_names),
                default_label="max",
            )

        if is_classification_dataset(dataset):
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
        time_mode=time_mode,
        video_step=video_step,
        video_interval=video_interval,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize local image and GeoTIFF datasets.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Folder containing supported dataset folders.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="cifar10", help="Dataset to visualize. Aliases include gtrsb, sturm, sen1, and floodcast.")
    parser.add_argument(
        "--split",
        choices=[
            "train",
            "test",
            "sentinel1",
            "sentinel2",
            "hand_s1",
            "hand_s2",
            "weak_s1",
            "weak_s2",
            "perm_water_s1",
            *FLOODCASTBENCH_SOURCES,
            "all",
        ],
        default=None,
        help="Data split/source/block to visualize. For FloodCastBench, use one of its block keys or all.",
    )
    parser.add_argument("--samples-per-class", type=int, default=8, help="Number of examples per class.")
    parser.add_argument("--class-name", help="Optional class name or class id to visualize for CIFAR/GTSRB datasets.")
    parser.add_argument("--max-classes", type=int, help="Limit the number of classes shown, useful for CIFAR-100 and GTSRB.")
    parser.add_argument("--columns", type=int, help="Number of image columns in the output grid.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used to pick samples.")
    parser.add_argument(
        "--time-mode",
        choices=["evenly_spaced", "video", "first", "middle", "last", "random"],
        default=None,
        help="FloodCastBench temporal selection mode. Defaults to evenly_spaced for temporal data and random for fixed data.",
    )
    parser.add_argument("--video-step", type=int, default=10, help="FloodCastBench video mode: display one frame every N files.")
    parser.add_argument("--video-interval", type=int, default=120, help="FloodCastBench video mode: delay between frames in milliseconds.")
    return parser.parse_args()


def build_args():
    if len(sys.argv) == 1:
        return interactive_args()

    args = parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    data_dir = dataset_dir(args.data_root, args.dataset)
    label_names = load_label_names(data_dir, args.dataset) if data_dir.exists() else []
    args.split = validate_split_for_dataset(args.dataset, args.split or default_split_for_dataset(args.dataset))

    if args.class_name and not is_classification_dataset(args.dataset):
        raise SystemExit("--class-name is only available for CIFAR/GTSRB classification datasets.")

    if dataset_kind(args.dataset) == "floodcastbench":
        is_temporal = is_floodcast_temporal_source(args.split)
        if args.time_mode is None:
            args.time_mode = "evenly_spaced" if is_temporal else "random"
        elif not is_temporal and args.time_mode in {"video", "first", "middle", "last"}:
            raise SystemExit("This FloodCastBench block is fixed in time; use no --time-mode or --time-mode random.")
        elif args.time_mode == "video" and not is_floodcast_video_source(args.split):
            raise SystemExit("--time-mode video is available only for FloodCastBench forecast/rainfall splits.")
    elif args.time_mode == "video":
        raise SystemExit("--time-mode video is only available for FloodCastBench.")
    elif args.time_mode is None:
        args.time_mode = "random"

    args.class_id = resolve_class_id(args.class_name, label_names) if args.class_name else None
    return args


def main():
    args = build_args()
    data_dir = dataset_dir(args.data_root, args.dataset)
    if not data_dir.exists():
        raise SystemExit(f"Dataset folder not found: {data_dir}")

    label_names = load_label_names(data_dir, args.dataset)
    images, labels, _ = load_split(data_dir, args.dataset, args.split)

    if dataset_kind(args.dataset) == "floodcastbench" and args.time_mode == "video":
        label_id = int(labels[0]) if len(labels) else 0
        plot_floodcast_video(images, label_names, label_id, args.split, args.video_step, args.video_interval)
        return

    if dataset_kind(args.dataset) == "floodcastbench":
        indices = select_floodcast_examples(labels, args.samples_per_class, args.seed, args.time_mode)
    else:
        indices = select_examples(labels, args.samples_per_class, args.seed, args.class_id)

    if args.max_classes and args.class_id is None:
        max_images = args.max_classes * args.samples_per_class
        indices = indices[:max_images]

    columns = args.columns or (args.samples_per_class if is_classification_dataset(args.dataset) else 1)
    plot_examples(images, labels, label_names, indices, columns)


if __name__ == "__main__":
    main()

