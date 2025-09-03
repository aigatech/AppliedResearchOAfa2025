# agent.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_classifier.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
LABELENC_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")
SAMPLES_PATH = os.path.join(MODEL_DIR, "sample_rows.joblib")
BACKGROUND_PATH = os.path.join(MODEL_DIR, "background.npy")  # saved scaled background for SHAP


class TCGATumorClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None

    def _infer_label_col(self, df: pd.DataFrame):
        for candidate in ["CANCER_TYPE", "Tumor_Type", "cancer_type", "label"]:
            if candidate in df.columns:
                return candidate
        return df.columns[-1]

    def prepare_xy(self, df: pd.DataFrame):
        df = df.copy()
        label_col = self._infer_label_col(df)

        # non-feature columns detection (drop obvious non-numeric columns except label)
        non_feature_cols = {label_col}
        for col in df.columns:
            if col == label_col:
                continue
            if df[col].dtype == object:
                sample_vals = df[col].dropna().head(8).astype(str).str.strip()
                convertible = True
                for v in sample_vals:
                    try:
                        float(v)
                    except:
                        convertible = False
                        break
                if not convertible:
                    non_feature_cols.add(col)

        feature_cols = [c for c in df.columns if c not in non_feature_cols]

        # convert features to numeric
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

        # drop rows with NaNs in features
        if df[feature_cols].isna().any(axis=None):
            df = df.dropna(subset=feature_cols)

        X = df[feature_cols].values.astype(float)
        y_raw = df[label_col].values
        self.feature_names = feature_cols
        return X, y_raw, label_col, feature_cols

    def fit(self, df: pd.DataFrame, save: bool = True, n_saved_examples: int = 7, background_size: int = 200):
        """
        Fit scaler + XGBoost classifier on DataFrame df.
        Persists model artifacts and a small scaled background for SHAP.
        """
        X, y_raw, label_col, feature_cols = self.prepare_xy(df)

        # encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y_raw)

        # scale features
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = self.scaler.fit_transform(X)

        # XGBoost classifier (CPU friendly)
        self.model = XGBClassifier(
            objective="multi:softprob" if len(np.unique(y)) > 2 else "binary:logistic",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss" if len(np.unique(y)) > 2 else "logloss",
            n_jobs=-1,
            tree_method="hist",
        )
        self.model.fit(X_scaled, y)

        # Save artifacts
        if save:
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            joblib.dump(self.label_encoder, LABELENC_PATH)
            joblib.dump(self.feature_names, FEATURES_PATH)
            print(f"Saved model artifacts to {MODEL_DIR}")

            # Save example rows (first n_saved_examples)
            sample_rows = []
            for i in range(min(n_saved_examples, X.shape[0])):
                sample_rows.append({
                    "values": X[i].tolist(),  # raw (unscaled) values for display
                    "label": self.label_encoder.inverse_transform([y[i]])[0]
                })
            joblib.dump(sample_rows, SAMPLES_PATH)
            print(f"Saved {len(sample_rows)} sample rows to {SAMPLES_PATH}")

            # Save a scaled background subset for SHAP (random sample of X_scaled)
            if background_size > 0:
                rng = np.random.default_rng(seed=42)
                n_background = min(background_size, X_scaled.shape[0])
                idx = rng.choice(X_scaled.shape[0], size=n_background, replace=False)
                background = X_scaled[idx, :]
                # Save as numpy .npy
                np.save(BACKGROUND_PATH, background)
                print(f"Saved SHAP background (shape {background.shape}) to {BACKGROUND_PATH}")

    def load(self):
        """
        Load model, scaler, label encoder, and feature names.
        """
        missing = []
        for p in [MODEL_PATH, SCALER_PATH, LABELENC_PATH, FEATURES_PATH]:
            if not os.path.exists(p):
                missing.append(p)
        if missing:
            raise FileNotFoundError(f"Missing model files: {missing}. Run training (main.py) first.")

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.label_encoder = joblib.load(LABELENC_PATH)
        self.feature_names = joblib.load(FEATURES_PATH)
        print("Loaded model and artifacts from disk.")

    def predict_row(self, values):
        if self.model is None or self.scaler is None or self.label_encoder is None or self.feature_names is None:
            self.load()

        arr = np.asarray(values, dtype=float).reshape(1, -1)
        if arr.shape[1] != len(self.feature_names):
            raise ValueError(f"Feature length mismatch: model expects {len(self.feature_names)}, got {arr.shape[1]}")

        Xs = self.scaler.transform(arr)
        pred_idx = self.model.predict(Xs)[0]
        return self.label_encoder.inverse_transform([pred_idx])[0]

    def predict_proba_row(self, values):
        if self.model is None or self.scaler is None or self.label_encoder is None or self.feature_names is None:
            self.load()

        arr = np.asarray(values, dtype=float).reshape(1, -1)
        if arr.shape[1] != len(self.feature_names):
            raise ValueError(f"Feature length mismatch: model expects {len(self.feature_names)}, got {arr.shape[1]}")

        Xs = self.scaler.transform(arr)
        probs = self.model.predict_proba(Xs)[0]
        return dict(zip(self.label_encoder.classes_, probs))

    def load_saved_samples(self):
        if os.path.exists(SAMPLES_PATH):
            return joblib.load(SAMPLES_PATH)
        return []

    def load_shap_background(self):
        if os.path.exists(BACKGROUND_PATH):
            return np.load(BACKGROUND_PATH)
        return None
