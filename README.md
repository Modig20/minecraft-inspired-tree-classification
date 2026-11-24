# Tree Leaf Classification

This project compares HOG + manual NumPy KNN, a randomly initialized custom CNN, and fine-tuned pretrained MobileNetV3-Small for multi-label tree-group classification. A species may map to more than one group; for example, White Oak belongs to both Oak and Pale Oak.

## Setup

```powershell
cd "C:\Users\ASUS\Documents\School related\Computer Vision\Com Vis Final"
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r computer_vision_project\requirements.txt
```

## Dataset

Download the Leafsnap dataset yourself and place it under `computer_vision_project\data\raw`. The folder can be organized by species. The Leafsnap subset contains these eight species:

- Quercus robur
- Quercus alba
- Quercus velutina
- Quercus rubra
- Picea abies
- Populus tremuloides
- Prunus yedoensis (Yoshino Cherry; Leafsnap label omits the hybrid marker)
- Prunus subhirtella

The other requested species are marked unavailable in the app rather than being fabricated. White Oak (`Quercus alba`) maps to both Oak and Pale Oak as requested.

The user-provided natural photos are stored under `data\additional_group_images\train\<Group>\` for Spruce, Oak, Dark Oak, Poplar, and Cherry. Additional photos can be added in the same layout. These supplemental images are training-only and are not included in validation or test metrics. Correct predictions on these same photos are training-sample checks, not evidence of generalization. For better performance on natural photos, add multiple independently labeled examples per group with varied leaves, backgrounds, lighting, and viewpoints, then evaluate on separate natural photos that were never used for training.

The script checks the dataset and exits clearly if any selected species has no images.

## Execution

Run from `computer_vision_project`:

```powershell
cd computer_vision_project
python training\prepare_dataset.py --raw-dir data\raw --processed-dir data\processed --seed 42
python training\train_classical.py --processed-dir data\processed --weights-dir weights --k 5
python training\train_cnn.py --processed-dir data\processed --weights-dir weights --epochs 20 --batch-size 32 --seed 42
python training\train_mobilenet.py --processed-dir data\processed --weights-dir weights --epochs 10 --batch-size 32 --seed 42
python evaluation\evaluate_models.py --processed-dir data\processed --weights-dir weights --results-dir results
python evaluation\generate_comparison.py --results-dir results
streamlit run app.py
```

The test split is not loaded by any training script and is used only in the final evaluation script. Models predict group membership, and a multi-label exact-match accuracy requires every predicted group for an image to match the ground truth.

To continue custom CNN training from its saved best checkpoint, add `--resume` and set `--epochs` to the number of additional epochs to run. For example:

```powershell
python training\train_cnn.py --processed-dir data\processed --weights-dir weights --epochs 6 --batch-size 32 --seed 42 --resume
```

Resuming restores the saved model weights and history; the optimizer is initialized again. Without `--resume`, the CNN starts from random weights.
