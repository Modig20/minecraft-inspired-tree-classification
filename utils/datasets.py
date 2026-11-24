from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.taxonomy import categories_for_species


class ImageFolderDataset(Dataset):
    def __init__(self, split_dir: Path, class_names, transform=None):
        self.split_dir = Path(split_dir)
        self.class_names = list(class_names)
        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}
        self.transform = transform
        self.samples = []
        for class_name in self.class_names:
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((path, self.class_to_index[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class GroupImageDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        species_names,
        group_names,
        transform=None,
        additional_group_dir=None,
    ):
        self.split_dir = Path(split_dir)
        self.species_names = list(species_names)
        self.group_names = list(group_names)
        self.group_to_index = {name: index for index, name in enumerate(self.group_names)}
        self.transform = transform
        self.samples = []

        for species in self.species_names:
            class_dir = self.split_dir / species
            if not class_dir.exists():
                continue
            target = self._target(categories_for_species(species))
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((path, target))

        if additional_group_dir is not None:
            additional_group_dir = Path(additional_group_dir)
            if additional_group_dir.exists():
                for group_name in self.group_names:
                    group_dir = additional_group_dir / group_name
                    if not group_dir.exists():
                        continue
                    target = self._target([group_name])
                    for path in sorted(group_dir.iterdir()):
                        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                            self.samples.append((path, target))

    def _target(self, categories):
        target = [0.0] * len(self.group_names)
        for category in categories:
            if category in self.group_to_index:
                target[self.group_to_index[category]] = 1.0
        if not any(target):
            raise ValueError(f"No supported group label found for {categories}.")
        return target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.float32)
