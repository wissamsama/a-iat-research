import pickle
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CIFAR10BatchDataset(Dataset):
    """Dataset PyTorch for the local CIFAR-10 python batch files."""

    def __init__(self, data_dir="data/CIFAR-10", train=True, transform=None):
        self.data_dir = Path(data_dir)
        self.train = train
        self.transform = transform

        if not self.data_dir.exists():
            raise FileNotFoundError(f"CIFAR-10 data folder not found: {self.data_dir}")

        batch_names = [f"data_batch_{index}" for index in range(1, 6)] if train else ["test_batch"]
        images = []
        labels = []

        for batch_name in batch_names:
            batch_path = self.data_dir / batch_name
            if not batch_path.exists():
                raise FileNotFoundError(f"Missing CIFAR-10 batch file: {batch_path}")

            batch = self._load_pickle(batch_path)
            batch_images = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            images.append(batch_images)
            labels.extend(batch["labels"])

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

    @staticmethod
    def _load_pickle(path):
        with path.open("rb") as file:
            return pickle.load(file, encoding="latin1")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total
