import argparse
import json
import hashlib
import random
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.taxonomy import DEFAULT_SPECIES

SUPPORTED = {".jpg", ".jpeg", ".png"}


def find_species_images(raw_dir, species):
    found = {name: [] for name in species}
    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        relative = str(path.relative_to(raw_dir)).lower().replace("_", " ").replace("-", " ")
        for name in species:
            normalized = name.lower().replace("_", " ").replace("-", " ")
            if normalized in relative or normalized.replace(" ", "") in relative.replace(" ", ""):
                found[name].append(path)
                break
    return found


def extract_selected_species(raw_dir, species):
    zip_candidates = []
    for directory in (
        raw_dir,
        raw_dir.parent,
        raw_dir.parent.parent / "data" / "raw",
        PROJECT_ROOT.parent / "data" / "raw",
        Path.cwd(),
        Path.cwd() / "data" / "raw",
    ):
        if directory.exists():
            zip_candidates.extend(sorted(directory.glob("*.zip")))
    zip_candidates = list(dict.fromkeys(zip_candidates))

    found = find_species_images(raw_dir, species)
    if all(found[name] for name in species):
        return

    for zip_path in zip_candidates:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                suffix = Path(entry.filename).suffix.lower()
                if suffix not in SUPPORTED:
                    continue
                normalized_entry = entry.filename.lower().replace("_", " ").replace("-", " ")
                matched_name = None
                for species_name in species:
                    normalized_species = species_name.lower().replace("_", " ").replace("-", " ")
                    if normalized_species in normalized_entry:
                        matched_name = species_name
                        break
                if matched_name is None:
                    continue
                species_folder = matched_name.replace(" ", "_")
                path_parts = Path(entry.filename).parts
                source_subfolder = ""
                for marker in ("field", "lab"):
                    if marker in path_parts:
                        source_subfolder = marker
                        break
                destination = raw_dir / "extracted" / species_folder / source_subfolder / Path(entry.filename).name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    with archive.open(entry) as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target)


def prepare(raw_dir, processed_dir, species, seed=42, train_ratio=0.70, val_ratio=0.15):
    random.seed(seed)
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_selected_species(raw_dir, species)

    found = find_species_images(raw_dir, species)
    missing = [name for name in species if not found[name]]
    if missing:
        raise RuntimeError(
            "Dataset images are missing for: " + ", ".join(missing)
            + ". Place folders matching these species under " + str(raw_dir)
        )

    for split in ("train", "val", "test"):
        split_dir = processed_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)

    counts = []
    for name in species:
        unique_paths = []
        seen_hashes = set()
        for path in found[name]:
            digest = hashlib.sha1(path.read_bytes()).hexdigest()
            if digest not in seen_hashes:
                seen_hashes.add(digest)
                unique_paths.append(path)
        paths = unique_paths
        random.shuffle(paths)
        train_end = int(len(paths) * train_ratio)
        val_end = train_end + int(len(paths) * val_ratio)
        splits = {"train": paths[:train_end], "val": paths[train_end:val_end], "test": paths[val_end:]}
        for split, split_paths in splits.items():
            destination = processed_dir / split / name
            destination.mkdir(parents=True, exist_ok=True)
            for source in split_paths:
                shutil.copy2(source, destination / source.name)
            counts.append({"species": name, "split": split, "count": len(split_paths)})

    processed_dir.mkdir(parents=True, exist_ok=True)
    with (processed_dir / "classes.json").open("w", encoding="utf-8") as handle:
        json.dump(species, handle, indent=2)
    pd.DataFrame(counts).to_csv(processed_dir / "dataset_counts.csv", index=False)
    print(pd.DataFrame(counts).pivot(index="species", columns="split", values="count").fillna(0).to_string())
    print(f"Prepared dataset at: {processed_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split Leafsnap images into train, validation, and test sets.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--species", nargs="*", default=DEFAULT_SPECIES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.raw_dir, args.processed_dir, args.species, args.seed)


if __name__ == "__main__":
    main()
