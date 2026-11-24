import argparse
import json
import time
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, multilabel_confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.classical_model import HOGKNNClassifier, extract_group_features_from_directory
from models.cnn_model import CustomCNN
from models.mobilenet_model import create_mobilenet
from utils.datasets import GroupImageDataset
from utils.taxonomy import available_categories


def metric_row(name, y_true, y_pred, training_time, parameter_count, extra=None):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "training_time_seconds": float(training_time),
        "parameter_count": parameter_count,
        **(extra or {}),
    }


def save_confusion_matrix(y_true, y_pred, class_names, path):
    matrices = multilabel_confusion_matrix(y_true, y_pred)
    columns = min(3, len(class_names))
    rows = (len(class_names) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes = np.asarray(axes).reshape(-1)
    output = {}
    for index, (class_name, matrix) in enumerate(zip(class_names, matrices)):
        axes[index].imshow(matrix, cmap="Blues")
        axes[index].set_xticks([0, 1], labels=["No", "Yes"])
        axes[index].set_yticks([0, 1], labels=["No", "Yes"])
        axes[index].set_xlabel("Predicted")
        axes[index].set_ylabel("Actual")
        axes[index].set_title(class_name)
        for row in range(2):
            for col in range(2):
                axes[index].text(col, row, str(matrix[row, col]), ha="center", va="center")
        output[class_name] = matrix.tolist()
    for axis in axes[len(class_names):]:
        axis.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return output


def evaluate(processed_dir, weights_dir, results_dir):
    processed_dir, weights_dir, results_dir = map(Path, (processed_dir, weights_dir, results_dir))
    class_file = processed_dir / "classes.json"
    if not class_file.exists():
        raise FileNotFoundError("Run prepare_dataset.py before evaluation.")
    with class_file.open("r", encoding="utf-8") as handle:
        species_names = json.load(handle)
    class_names = available_categories(species_names)
    results_dir.mkdir(parents=True, exist_ok=True)

    test_features, test_labels = extract_group_features_from_directory(
        processed_dir / "test", species_names, class_names
    )
    test_labels = test_labels.astype(np.int64)
    comparison = []
    confusion = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mobile_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    classical_path = weights_dir / "hog_knn.npz"
    if classical_path.exists():
        classical = HOGKNNClassifier.load(classical_path)
        if classical.class_names != class_names:
            raise RuntimeError("HOG + KNN group labels do not match the dataset taxonomy; retrain it.")
        start = time.perf_counter()
        classical_predictions, _ = classical.predict(test_features)
        elapsed = time.perf_counter() - start
        row = metric_row("HOG + KNN", test_labels, classical_predictions,
                         json.loads((weights_dir / "hog_knn_metadata.json").read_text())["training_time_seconds"],
                         None, {"stored_training_samples": len(classical.train_labels),
                                 "feature_representation_size": classical.feature_size,
                                 "prediction_time_per_image_seconds": elapsed / len(test_labels)})
        comparison.append(row)
        confusion["HOG + KNN"] = save_confusion_matrix(
            test_labels, classical_predictions, class_names, results_dir / "confusion_hog_knn.png")

    cnn_test_set = GroupImageDataset(processed_dir / "test", species_names, class_names, cnn_transform)
    cnn_test_loader = DataLoader(cnn_test_set, batch_size=32, shuffle=False, num_workers=0)
    mobile_test_set = GroupImageDataset(processed_dir / "test", species_names, class_names, mobile_transform)
    mobile_test_loader = DataLoader(mobile_test_set, batch_size=32, shuffle=False, num_workers=0)

    cnn_path = weights_dir / "custom_cnn.pth"
    if cnn_path.exists():
        checkpoint = torch.load(cnn_path, map_location=device, weights_only=False)
        if checkpoint.get("class_names") != class_names:
            raise RuntimeError("Custom CNN group labels do not match the dataset taxonomy; retrain it.")
        cnn = CustomCNN(len(class_names)).to(device)
        cnn.load_state_dict(checkpoint["model_state"])
        cnn.eval()
        predictions, start = [], time.perf_counter()
        with torch.no_grad():
            for images, _ in cnn_test_loader:
                predictions.extend((torch.sigmoid(cnn(images.to(device))) >= 0.5).cpu().numpy().astype(np.int64))
        elapsed = time.perf_counter() - start
        metadata = json.loads((weights_dir / "custom_cnn_metadata.json").read_text())
        row = metric_row("Custom CNN", test_labels, predictions,
                         metadata["training_time_seconds"], metadata["parameter_count"],
                         {"prediction_time_per_image_seconds": elapsed / len(test_labels)})
        comparison.append(row)
        confusion["Custom CNN"] = save_confusion_matrix(
            test_labels, predictions, class_names, results_dir / "confusion_custom_cnn.png")

    mobile_path = weights_dir / "mobilenet_v3_small.pth"
    if mobile_path.exists():
        checkpoint = torch.load(mobile_path, map_location=device, weights_only=False)
        if checkpoint.get("class_names") != class_names:
            raise RuntimeError("MobileNet group labels do not match the dataset taxonomy; retrain it.")
        mobile = create_mobilenet(len(class_names), pretrained=False).to(device)
        mobile.load_state_dict(checkpoint["model_state"])
        mobile.eval()
        predictions, start = [], time.perf_counter()
        with torch.no_grad():
            for images, _ in mobile_test_loader:
                predictions.extend((torch.sigmoid(mobile(images.to(device))) >= 0.5).cpu().numpy().astype(np.int64))
        elapsed = time.perf_counter() - start
        metadata = json.loads((weights_dir / "mobilenet_metadata.json").read_text())
        row = metric_row("MobileNetV3-Small", test_labels, predictions,
                         metadata["training_time_seconds"], metadata["parameter_count"],
                         {"prediction_time_per_image_seconds": elapsed / len(test_labels),
                          "weights_source": metadata["weights_source"]})
        comparison.append(row)
        confusion["MobileNetV3-Small"] = save_confusion_matrix(
            test_labels, predictions, class_names, results_dir / "confusion_mobilenet.png")

    if not comparison:
        raise RuntimeError("No trained model weights were found in " + str(weights_dir))

    output = {
        "class_names": class_names,
        "species_names": species_names,
        "prediction_type": "multi-label categories",
        "models": comparison,
        "confusion_matrices": confusion,
    }
    with (results_dir / "evaluation_results.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    evaluate(args.processed_dir, args.weights_dir, args.results_dir)


if __name__ == "__main__":
    main()
