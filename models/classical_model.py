from pathlib import Path

import numpy as np

from utils.preprocessing import extract_hog_features, load_rgb_image


class HOGKNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.train_features = None
        self.train_labels = None
        self.class_names = None
        self.feature_size = None

    def fit(self, features, labels, class_names):
        self.train_features = np.ascontiguousarray(features, dtype=np.float32)
        self.train_labels = np.asarray(labels, dtype=np.float32)
        if self.train_labels.ndim != 2 or self.train_labels.shape[1] != len(class_names):
            raise ValueError("Labels must be a multi-label matrix matching class_names.")
        self.class_names = list(class_names)
        self.feature_size = self.train_features.shape[1]

    def predict_one(self, feature):
        distances = np.sqrt(np.sum((self.train_features - feature) ** 2, axis=1))
        neighbor_count = min(self.k, len(distances))
        neighbor_indices = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
        neighbor_labels = self.train_labels[neighbor_indices]
        votes = np.sum(neighbor_labels, axis=0).astype(np.int64)
        predicted = votes > (neighbor_count / 2)
        if not predicted.any():
            predicted[np.argmax(votes)] = True
        vote_counts = {self.class_names[i]: int(votes[i]) for i in range(len(self.class_names))}
        return predicted, vote_counts

    def predict(self, features):
        predictions, votes = [], []
        for feature in features:
            label, vote_counts = self.predict_one(feature)
            predictions.append(label)
            votes.append(vote_counts)
        return np.asarray(predictions, dtype=np.int64), votes

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, train_features=self.train_features,
                            train_labels=self.train_labels,
                            class_names=np.asarray(self.class_names),
                            k=np.asarray(self.k))

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        model = cls(k=int(data["k"]))
        model.train_features = data["train_features"]
        model.train_labels = data["train_labels"]
        model.class_names = data["class_names"].tolist()
        model.feature_size = model.train_features.shape[1]
        return model


def extract_features_from_directory(split_dir, class_names):
    features, labels = [], []
    split_dir = Path(split_dir)
    for label, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                features.append(extract_hog_features(load_rgb_image(path)))
                labels.append(label)
    if not features:
        raise ValueError(f"No supported images found in {split_dir}")
    return np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.int64)


def extract_group_features_from_directory(
    split_dir, species_names, group_names, additional_group_dir=None
):
    from utils.taxonomy import categories_for_species

    features, labels = [], []
    split_dir = Path(split_dir)
    group_to_index = {name: index for index, name in enumerate(group_names)}
    for species in species_names:
        class_dir = split_dir / species
        if not class_dir.exists():
            continue
        target = np.zeros(len(group_names), dtype=np.float32)
        for category in categories_for_species(species):
            if category in group_to_index:
                target[group_to_index[category]] = 1.0
        if not target.any():
            raise ValueError(f"No supported group label found for species {species!r}.")
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                features.append(extract_hog_features(load_rgb_image(path)))
                labels.append(target.copy())
    if additional_group_dir is not None:
        additional_group_dir = Path(additional_group_dir)
        if additional_group_dir.exists():
            for group_name in group_names:
                group_dir = additional_group_dir / group_name
                if not group_dir.exists():
                    continue
                target = np.zeros(len(group_names), dtype=np.float32)
                target[group_to_index[group_name]] = 1.0
                for path in sorted(group_dir.iterdir()):
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        features.append(extract_hog_features(load_rgb_image(path)))
                        labels.append(target.copy())
    if not features:
        raise ValueError(f"No supported images found in {split_dir}")
    return np.stack(features).astype(np.float32), np.stack(labels).astype(np.float32)
