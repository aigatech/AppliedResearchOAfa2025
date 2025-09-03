import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image  # <-- added
import gradio as gr
from agent import TCGATumorClassifier
from nlp_agent import CancerInfoAgent

# optional warnings mute
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Try import shap
try:
    import shap
    HAS_SHAP = True
except Exception as e:
    shap = None
    HAS_SHAP = False
    print("shap not installed or failed to import:", e)

DATA_PATH = "data/mini_tcga.csv"
MODEL_DIR = "models"

# Constants
TOP_K_GLOBAL = 20   # number of most important features to show globally
TOP_K_LOCAL = 10    # number of most important features to show per prediction
MAX_FEATURES_FOR_SHAP = 500  # cap to keep SHAP fast


# Load classifier
clf = TCGATumorClassifier()
try:
    clf.load()
except FileNotFoundError as e:
    raise FileNotFoundError("Saved model artifacts not found. Run `python main.py` first to train and save the model.") from e

# Load saved example rows
saved_samples = clf.load_saved_samples()
example_texts = []
for i, s in enumerate(saved_samples):
    # join values as space-separated with 6 decimals
    vals = " ".join([f"{v:.6f}" for v in s["values"]])
    example_texts.append((f"Sample #{i+1} (true: {s.get('label','?')})", vals))

nlp = CancerInfoAgent()

# helper: convert fig to pil
def fig_to_pil(fig, fmt="png"):
    """
    Convert a matplotlib figure to a PIL.Image and close the figure.
    Return: PIL.Image.Image
    """
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGBA")
    plt.close(fig)
    return pil_img


# plot global feature importance (bar chart)
def plot_global_feature_importance(clf, top_k=TOP_K_GLOBAL):
    if clf.model is None or clf.feature_names is None:
        clf.load()
    fi = getattr(clf.model, "feature_importances_", None)
    if fi is None:
        raise RuntimeError("Model has no feature_importances_. Cannot plot global importance.")

    fi = np.array(fi)
    idx = np.argsort(fi)[::-1][:top_k]
    names = [clf.feature_names[i] for i in idx]
    vals = fi[idx]

    fig, ax = plt.subplots(figsize=(6, max(3, top_k * 0.18)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals[::-1], align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Feature importance (gain / normalized)")
    ax.set_title(f"Top {top_k} features by XGBoost importance")
    ax.invert_yaxis()
    return fig_to_pil(fig)
def plot_local_shap_contributions(clf, values, top_k=TOP_K_LOCAL, max_features_for_shap=MAX_FEATURES_FOR_SHAP):
    if shap is None:
        # Fallback: plot top feature_importances_ for features where input value is high
        fi = getattr(clf.model, "feature_importances_", None)
        if fi is None:
            raise RuntimeError("Neither shap nor feature_importances_ available to explain model.")
        arr = np.asarray(values, dtype=float).reshape(1, -1)
        idx = np.argsort(fi)[::-1][:top_k]
        names = [clf.feature_names[i] for i in idx]
        vals = arr[0, idx]
        fig, ax = plt.subplots(figsize=(6, max(3, top_k * 0.18)))
        ax.barh(range(len(names)), vals[::-1])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names[::-1], fontsize=8)
        ax.set_title("Top features (fallback visualization)")
        return fig_to_pil(fig)

    # Ensure model/scaler/feature_names loaded
    if clf.model is None or clf.scaler is None or clf.feature_names is None:
        clf.load()

    arr = np.asarray(values, dtype=float).reshape(1, -1)
    if arr.shape[1] != len(clf.feature_names):
        raise ValueError(f"Feature length mismatch: model expects {len(clf.feature_names)} features, got {arr.shape[1]}")

    # Scale input
    Xs = clf.scaler.transform(arr)

    # Select subset of features for SHAP speed
    fi = getattr(clf.model, "feature_importances_", None)
    fi = np.array(fi)
    selected_idx = np.argsort(fi)[::-1][:min(max_features_for_shap, len(fi))]

    # Prepare reduced input for SHAP
    Xs_zeroed = np.zeros_like(Xs)
    Xs_zeroed[:, selected_idx] = Xs[:, selected_idx]

    # Compute SHAP
    explainer = shap.TreeExplainer(clf.model)
    shap_values = explainer(Xs_zeroed)  # returns array of shape (n_samples, n_features, n_classes) or list

    # Select shap values for predicted class
    pred_idx = int(clf.model.predict(Xs)[0])
    if isinstance(shap_values, list):
        class_shap = shap_values[pred_idx].values[0]  # take first (and only) row
    else:
        # SHAP array: (n_samples, n_features, n_classes) -> pick sample 0 and pred_idx
        if shap_values.values.ndim == 3:
            class_shap = shap_values.values[0, :, pred_idx]
        else:
            class_shap = shap_values.values[0, :]  # fallback if 2D

    # Build contributions
    contributions = []
    for i_local, feat_i in enumerate(selected_idx):
        fname = clf.feature_names[feat_i]
        fval = float(arr[0, feat_i])
        s = float(class_shap[i_local])
        contributions.append({"feature": fname, "value": fval, "shap": s, "abs_shap": abs(s)})

    # Sort top K
    contributions_sorted = sorted(contributions, key=lambda x: x["abs_shap"], reverse=True)[:top_k]
    names = [c["feature"] for c in contributions_sorted]
    shap_vals = [c["shap"] for c in contributions_sorted]

    # Plot
    fig, ax = plt.subplots(figsize=(6, max(3, len(names)*0.18)))
    y_pos = np.arange(len(names))
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in shap_vals]
    ax.barh(y_pos, shap_vals[::-1], color=colors[::-1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("SHAP value (contribution to predicted class)")
    ax.set_title(f"Top {len(names)} contributing features (SHAP) — predicted class index {pred_idx}")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.invert_yaxis()
    return fig_to_pil(fig)



def parse_input_to_values(sample_text):
    if sample_text is None:
        return []
    tokens = sample_text.strip().replace(",", " ").split()
    values = []
    for t in tokens:
        try:
            values.append(float(t))
        except:
            continue
    return values


def format_probs_md(probs_dict):
    sorted_items = sorted(probs_dict.items(), key=lambda x: -x[1])
    md = "\n".join([f"- **{k}**: {v:.3f}" for k, v in sorted_items])
    return md


# Build Gradio UI with controls for SHAP settings
with gr.Blocks() as demo:
    gr.Markdown("# TCGA Tumor Classifier + SHAP Visuals")
    gr.Markdown("Paste a single row of numeric gene-expression values (space- or comma-separated). Or choose an example sample below.")

    with gr.Row():
        example_dropdown_choices = [t for t, _ in example_texts] if example_texts else []
        example_dropdown = gr.Dropdown(choices=example_dropdown_choices, label="Example samples", interactive=True)
        input_box = gr.Textbox(lines=4, placeholder="Paste a single row of numeric gene expression values (space- or comma-separated).")

    with gr.Row():
        load_button = gr.Button("Load Example into Input")
        predict_button = gr.Button("Predict & Explain")

    with gr.Row():
        max_feat_slider = gr.Slider(minimum=50, maximum=min(2000, len(clf.feature_names)), value=500, step=50, label="Max features considered for SHAP (speed vs completeness)")
        topk_slider = gr.Slider(minimum=5, maximum=200, value=30, step=1, label="Top K features to display")
    with gr.Row():
        out_md = gr.Markdown("")
    with gr.Row():
        global_img = gr.Image(label="Global feature importance", type="pil")
        local_img = gr.Image(label="Per-sample SHAP contribution / fallback", type="pil")

    # wiring
    def load_selected(choice_label):
        for title, vals in example_texts:
            if title == choice_label:
                return vals
        return ""

    def on_predict(sample_text, max_features_for_shap, top_k):
        values = parse_input_to_values(sample_text)
        if len(values) == 0:
            return "No numeric values found. Paste a row of numeric gene-values.", None, None

        expected_n = len(clf.feature_names)
        if len(values) == expected_n + 1:
            values = values[:-1]

        if len(values) != expected_n:
            return f"Expected {expected_n} features, but got {len(values)}.", None, None

        # predict
        pred_label = clf.predict_row(values)
        probs = clf.predict_proba_row(values)
        probs_md = format_probs_md(probs)

        # LLM explanation
        try:
            explanation = nlp.get_cancer_info(pred_label)
        except Exception as e:
            explanation = f"(LLM error: {e})"

        md = f"## Predicted tumor type: **{pred_label}**\n\n**Class probabilities:**\n{probs_md}\n\n---\n### About {pred_label}\n{explanation}\n\n*Disclaimer: Not medical advice.*"

        # plots
        try:
            global_png = plot_global_feature_importance(clf, top_k=30)
        except Exception as e:
            global_png = None
            print("Global plot error:", e)

        try:
           local_png = plot_local_shap_contributions(clf, values, max_features_for_shap=int(max_features_for_shap), top_k=int(top_k))

        except Exception as e:
            local_png = None
            print("Local SHAP error:", e)

        return md, global_png, local_png

    if example_texts:
        example_dropdown.change(fn=load_selected, inputs=example_dropdown, outputs=input_box)
        load_button.click(fn=load_selected, inputs=example_dropdown, outputs=input_box)

    predict_button.click(fn=on_predict, inputs=[input_box, max_feat_slider, topk_slider], outputs=[out_md, global_img, local_img])

if __name__ == "__main__":
    demo.launch()
