import argparse
import json
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_model import CustomCNN
from utils.datasets import GroupImageDataset
from utils.taxonomy import available_categories


def accuracy(outputs, labels):
    return (outputs.argmax(dim=1) == labels).float().mean().item()


def train(processed_dir, weights_dir, epochs=20, batch_size=32, seed=42, resume=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    processed_dir, weights_dir = Path(processed_dir), Path(weights_dir)
    with (processed_dir / "classes.json").open("r", encoding="utf-8") as handle:
        species_names = json.load(handle)
    group_names = available_categories(species_names)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    extra_train_dir = processed_dir.parent / "additional_group_images" / "train"
    train_set = GroupImageDataset(
        processed_dir / "train", species_names, group_names, transform, extra_train_dir
    )
    val_set = GroupImageDataset(processed_dir / "val", species_names, group_names, val_transform)
    if not train_set or not val_set:
        raise RuntimeError("Training or validation split is empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomCNN(len(group_names)).to(device)
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = weights_dir / "custom_cnn.pth"
    targets = np.asarray([target for _, target in train_set.samples], dtype=np.float32)
    positive_counts = targets.sum(axis=0)
    positive_weights = torch.tensor(
        np.clip((len(targets) - positive_counts) / np.maximum(positive_counts, 1), 1.0, 5.0),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_epoch = 0
    last_epoch = 0
    best_model_state = None
    previous_training_time = 0.0
    metadata_path = weights_dir / "custom_cnn_metadata.json"
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError("Cannot resume because the custom CNN checkpoint is missing.")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("class_names") != group_names or checkpoint.get("species_names") != species_names:
            raise RuntimeError("Cannot resume a CNN checkpoint with different class or species labels.")
        model.load_state_dict(checkpoint["model_state"])
        history = checkpoint.get("history", history)
        best_val_loss = checkpoint.get(
            "best_val_loss", min(history["val_loss"], default=float("inf"))
        )
        best_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch", 0))
        last_epoch = checkpoint.get("last_epoch", checkpoint.get("epoch", 0))
        best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in checkpoint["model_state"].items()
        }
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                previous_metadata = json.load(handle)
            if (previous_metadata.get("class_names") == group_names
                    and previous_metadata.get("species_names") == species_names):
                previous_training_time = previous_metadata.get("training_time_seconds", 0.0)

    start_epoch = last_epoch + 1
    end_epoch = last_epoch + epochs
    start = time.perf_counter()
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = train_correct = train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_correct += ((torch.sigmoid(outputs) >= 0.5) == (labels >= 0.5)).all(dim=1).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += ((torch.sigmoid(outputs) >= 0.5) == (labels >= 0.5)).all(dim=1).sum().item()
                val_total += labels.size(0)

        train_loss /= train_total
        val_loss /= val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        current_epoch = start_epoch + epoch - 1
        print(f"Epoch {current_epoch}/{end_epoch} - train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = current_epoch
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        last_epoch = current_epoch
        torch.save({
            "model_state": best_model_state,
            "class_names": group_names,
            "species_names": species_names,
            "history": history,
            "epoch": best_epoch,
            "best_epoch": best_epoch,
            "last_epoch": last_epoch,
            "best_val_loss": best_val_loss,
        }, checkpoint_path)

    training_time = previous_training_time + time.perf_counter() - start
    metadata = {
        "method": "Custom CNN",
        "weights_source": "random initialization",
        "training_time_seconds": training_time,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "epochs_trained": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "class_names": group_names,
        "species_names": species_names,
        "device": str(device),
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    with (weights_dir / "custom_cnn_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.processed_dir, args.weights_dir, args.epochs, args.batch_size, args.seed, args.resume)


if __name__ == "__main__":
    main()
