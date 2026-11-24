import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.classical_model import HOGKNNClassifier
from models.cnn_model import CustomCNN
from models.mobilenet_model import create_mobilenet
from utils.preprocessing import SUPPORTED_EXTENSIONS, extract_hog_features, load_rgb_image
from utils.prediction import load_classes
from utils.taxonomy import available_categories, dataset_category_coverage

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RESULTS_DIR = PROJECT_ROOT / "results"


@st.cache_resource
def load_models(cache_key):
    species_names = load_classes(PROCESSED_DIR)
    group_names = available_categories(species_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {"class_names": group_names, "device": device}

    classical_path = WEIGHTS_DIR / "hog_knn.npz"
    if classical_path.exists():
        classical = HOGKNNClassifier.load(classical_path)
        if classical.class_names != group_names:
            raise RuntimeError("HOG + KNN uses old labels. Retrain it for group classification.")
        models["classical"] = classical

    cnn_path = WEIGHTS_DIR / "custom_cnn.pth"
    if cnn_path.exists():
        checkpoint = torch.load(cnn_path, map_location=device, weights_only=False)
        if checkpoint.get("class_names") != group_names:
            raise RuntimeError("Custom CNN uses old labels. Retrain it for group classification.")
        cnn = CustomCNN(len(group_names)).to(device)
        cnn.load_state_dict(checkpoint["model_state"])
        cnn.eval()
        models["cnn"] = cnn

    mobile_path = WEIGHTS_DIR / "mobilenet_v3_small.pth"
    if mobile_path.exists():
        checkpoint = torch.load(mobile_path, map_location=device, weights_only=False)
        if checkpoint.get("class_names") != group_names:
            raise RuntimeError("MobileNet uses old labels. Retrain it for group classification.")
        mobile = create_mobilenet(len(group_names), pretrained=False).to(device)
        mobile.load_state_dict(checkpoint["model_state"])
        mobile.eval()
        models["mobilenet"] = mobile
    return models


def model_cache_key():
    paths = [
        PROCESSED_DIR / "classes.json",
        WEIGHTS_DIR / "hog_knn.npz",
        WEIGHTS_DIR / "custom_cnn.pth",
        WEIGHTS_DIR / "mobilenet_v3_small.pth",
    ]
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths
        if path.exists()
    )


def predict_neural(model, image, device, input_size):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        scores = torch.sigmoid(model(tensor))[0].cpu().numpy()
    elapsed = time.perf_counter() - start
    predicted = scores >= 0.5
    if not predicted.any():
        predicted[int(np.argmax(scores))] = True
    return predicted, scores, elapsed


st.set_page_config(page_title="Tree Leaf Classification", layout="wide")
st.title("Tree Leaf Classification")

st.write(
    "Compare three multi-label tree-group classifiers: manually implemented HOG + KNN, "
    "a custom PyTorch CNN initialized from random weights, and a fine-tuned "
    "MobileNetV3-Small pretrained on ImageNet."
)

tab_upload, tab_compare = st.tabs(["Image Upload", "Model Comparison"])

with tab_upload:
    st.subheader("Upload a leaf image")
    st.caption("Accepted formats: JPG, JPEG, and PNG")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            st.error("Unsupported file format. Upload a JPG, JPEG, or PNG image.")
        else:
            try:
                image = load_rgb_image(uploaded)
                st.image(image, caption="Uploaded leaf image", use_container_width=True)
            except Exception as exc:
                st.error(f"Could not read the uploaded image: {exc}")
            else:
                try:
                    loaded = load_models(model_cache_key())
                except Exception as exc:
                    st.error(str(exc))
                else:
                    group_names = loaded["class_names"]
                    device = loaded["device"]
                    st.info(
                        "These are group-level predictions. Only groups represented in the selected "
                        "Leafsnap species have training examples. Neural-network scores are not calibrated."
                    )
                    columns = st.columns(3)
                    model_predictions = {}

                    if "classical" in loaded:
                        start = time.perf_counter()
                        feature = extract_hog_features(image)
                        predicted, votes = loaded["classical"].predict_one(feature)
                        elapsed = time.perf_counter() - start
                        predicted_groups = tuple(
                            group_names[i] for i, is_predicted in enumerate(predicted) if is_predicted
                        )
                        model_predictions["HOG + KNN"] = predicted_groups
                        sorted_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
                        with columns[0]:
                            st.subheader("HOG + KNN")
                            st.success(", ".join(predicted_groups) if predicted_groups else "No group predicted")
                            st.write("K-nearest-neighbor votes:")
                            st.dataframe(
                                pd.DataFrame(sorted_votes, columns=["Group", "Neighbor votes"]),
                                hide_index=True,
                            )
                            st.caption(f"{elapsed * 1000:.2f} ms; votes are not calibrated probabilities")
                    else:
                        columns[0].warning("HOG + KNN weights are missing.")

                    if "cnn" in loaded:
                        predicted, scores, elapsed = predict_neural(loaded["cnn"], image, device, 128)
                        predicted_groups = tuple(
                            group_names[i] for i, is_predicted in enumerate(predicted) if is_predicted
                        )
                        model_predictions["Custom CNN"] = predicted_groups
                        with columns[1]:
                            st.subheader("Custom CNN")
                            st.success(", ".join(predicted_groups) if predicted_groups else "No group predicted")
                            st.metric("Highest group score", f"{scores.max():.2%}")
                            top_indices = np.argsort(scores)[::-1][:3]
                            st.dataframe(
                                pd.DataFrame({
                                    "Group": [group_names[i] for i in top_indices],
                                    "Model score": [f"{scores[i]:.2%}" for i in top_indices],
                                }),
                                hide_index=True,
                            )
                            st.caption(f"{elapsed * 1000:.2f} ms")
                    else:
                        columns[1].warning("Custom CNN checkpoint is missing.")

                    if "mobilenet" in loaded:
                        predicted, scores, elapsed = predict_neural(
                            loaded["mobilenet"], image, device, 224
                        )
                        predicted_groups = tuple(
                            group_names[i] for i, is_predicted in enumerate(predicted) if is_predicted
                        )
                        model_predictions["MobileNetV3-Small"] = predicted_groups
                        with columns[2]:
                            st.subheader("MobileNetV3-Small")
                            st.success(", ".join(predicted_groups) if predicted_groups else "No group predicted")
                            st.metric("Highest group score", f"{scores.max():.2%}")
                            top_indices = np.argsort(scores)[::-1][:3]
                            st.dataframe(
                                pd.DataFrame({
                                    "Group": [group_names[i] for i in top_indices],
                                    "Model score": [f"{scores[i]:.2%}" for i in top_indices],
                                }),
                                hide_index=True,
                            )
                            st.caption(f"{elapsed * 1000:.2f} ms")
                    else:
                        columns[2].warning("MobileNetV3-Small checkpoint is missing.")

                    if len(set(model_predictions.values())) > 1:
                        st.warning(
                            "The models disagree on this image. Treat the results as uncertain; "
                            "a high model score does not guarantee a correct group identification."
                        )

with tab_compare:
    st.subheader("Test-set model comparison")
    try:
        current_classes = load_classes(PROCESSED_DIR)
        coverage = pd.DataFrame(dataset_category_coverage(current_classes))
        st.caption("Partial or unavailable categories have missing species and should not be treated as well-covered.")
        st.dataframe(coverage, hide_index=True)
    except FileNotFoundError:
        st.warning("Dataset class list is missing. Prepare the dataset before training.")

    evaluation_path = RESULTS_DIR / "evaluation_results.json"
    if not evaluation_path.exists():
        st.warning("No evaluation results are available. Run dataset preparation, training, and evaluation.")
    else:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        table = pd.DataFrame(evaluation["models"])
        st.dataframe(table, hide_index=True)

        chart_names = {
            "accuracy_comparison.png": "Accuracy",
            "metrics_comparison.png": "Precision, recall, and F1",
            "prediction_speed_comparison.png": "Prediction speed",
        }
        for filename, caption in chart_names.items():
            chart = RESULTS_DIR / filename
            if chart.exists():
                st.image(str(chart), caption=caption)

        st.subheader("Confusion matrices")
        for model_name in evaluation["confusion_matrices"]:
            filename = {
                "HOG + KNN": "confusion_hog_knn.png",
                "Custom CNN": "confusion_custom_cnn.png",
                "MobileNetV3-Small": "confusion_mobilenet.png",
            }[model_name]
            chart = RESULTS_DIR / filename
            if chart.exists():
                st.image(str(chart), caption=model_name)

        if table.empty:
            st.warning("The evaluation table is empty; no recommendation can be made.")
        else:
            best = table.sort_values(["accuracy", "f1_score"], ascending=False).iloc[0]
            st.subheader("Recommended model")
            st.success(best["model"])
            st.write(
                f"{best['model']} is recommended because it has the highest measured test accuracy "
                f"({best['accuracy']:.2%}) and an F1-score of {best['f1_score']:.2%} "
                "among the evaluated models."
            )
