import argparse
import json
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.classical_model import HOGKNNClassifier, extract_group_features_from_directory
from utils.taxonomy import available_categories


def train(processed_dir, weights_dir, k=5):
    processed_dir, weights_dir = Path(processed_dir), Path(weights_dir)
    class_file = processed_dir / "classes.json"
    if not class_file.exists():
        raise FileNotFoundError("classes.json missing. Run prepare_dataset.py first.")
    with class_file.open("r", encoding="utf-8") as handle:
        species_names = json.load(handle)
    class_names = available_categories(species_names)
    start = time.perf_counter()
    extra_train_dir = processed_dir.parent / "additional_group_images" / "train"
    train_features, train_labels = extract_group_features_from_directory(
        processed_dir / "train", species_names, class_names, extra_train_dir
    )
    model = HOGKNNClassifier(k=k)
    model.fit(train_features, train_labels, class_names)
    training_time = time.perf_counter() - start

    weights_dir.mkdir(parents=True, exist_ok=True)
    model.save(weights_dir / "hog_knn.npz")
    npz_path = weights_dir / "hog_knn.npz"
    metadata = {
        "method": "HOG + KNN",
        "k": k,
        "training_time_seconds": training_time,
        "stored_training_samples": len(train_labels),
        "feature_representation_size": int(train_features.shape[1]),
        "class_names": class_names,
        "species_names": species_names,
        "model_path": str(npz_path),
    }
    with (weights_dir / "hog_knn_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))
    return model, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    train(args.processed_dir, args.weights_dir, args.k)


if __name__ == "__main__":
    main()
