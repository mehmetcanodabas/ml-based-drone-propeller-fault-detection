
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT_FILE = ROOT / "model_training" / "features.csv"
MODEL_FILE = ROOT / "model_training" / "gaussian_svm.joblib"

RESULTS_DIR = ROOT / "docs" / "results"

# ============================================================
# MODEL CONFIGURATION
# ============================================================

SPEEDS = (1, 2, 3)
FOLDS = (1, 2, 3)

FEATURE_NAMES = (
    "crest_factor",
    "kurtosis",
    "low_peak_frequency_hz",
    "autocorrelation_peak",
)

FEATURE_COLUMNS = [
    f"{channel}__{feature}"
    for channel in ("C1", "C2")
    for feature in FEATURE_NAMES
]

C = 1.0
GAMMA = 1.0


def create_model():

    return make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=C,
            gamma=GAMMA,
            class_weight="balanced",
        ),
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_data():

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Missing feature file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "speed",
        "condition",
        "record_id",
        "label",
        "fold",
        "window_index",
    } | set(FEATURE_COLUMNS)

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if len(df) != 1080:
        raise ValueError(
            f"Expected 1080 windows, found {len(df)}"
        )

    if df["record_id"].nunique() != 36:
        raise ValueError(
            "Expected 36 physical recordings"
        )

    if set(df["speed"]) != set(SPEEDS):
        raise ValueError(
            "Expected speeds 1, 2, 3"
        )

    if set(df["fold"]) != set(FOLDS):
        raise ValueError(
            "Expected folds 1, 2, 3"
        )

    if set(df["label"]) != {0, 1}:
        raise ValueError(
            "Expected labels 0 and 1"
        )

    if df.duplicated(
        ["record_id", "window_index"]
    ).any():
        raise ValueError(
            "Duplicate recording/window combination"
        )

    if not df.groupby("record_id").size().eq(30).all():
        raise ValueError(
            "Expected 30 windows per recording"
        )

    if not df.groupby("record_id")[
        ["speed", "condition", "fold", "label"]
    ].nunique().eq(1).all().all():
        raise ValueError(
            "Inconsistent recording metadata"
        )

    # All speeds use the same fold mapping.
    expected_conditions = {
        1: {"F0-R1", "F1-SV1", "F2-SV1", "F3-SV1"},
        2: {"F0-R2", "F1-SV2", "F2-SV2", "F3-SV2"},
        3: {"F0-R3", "F1-SV3", "F2-SV3", "F3-SV3"},
    }

    for speed in SPEEDS:

        speed_df = df[df["speed"] == speed]

        if speed_df["record_id"].nunique() != 12:
            raise ValueError(
                f"SP{speed}: expected 12 recordings"
            )

        for fold in FOLDS:

            conditions = set(
                speed_df.loc[
                    speed_df["fold"] == fold,
                    "condition"
                ]
            )

            if conditions != expected_conditions[fold]:
                raise ValueError(
                    f"Unexpected condition mapping "
                    f"for SP{speed}, fold {fold}: {conditions}"
                )

    values = df[FEATURE_COLUMNS].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Feature matrix contains NaN or infinity"
        )

    return df.sort_values(
        ["speed", "condition", "window_index"]
    ).reset_index(drop=True)


# ============================================================
# CROSS-VALIDATION
# ============================================================

def evaluate_cross_validation(df):

    window_results = []
    record_results = []

    for fold in FOLDS:

        train = df[df["fold"] != fold].copy()
        test = df[df["fold"] == fold].copy()

        train_ids = set(train["record_id"])
        test_ids = set(test["record_id"])

        if train_ids & test_ids:
            raise ValueError(
                "Recording leakage detected"
            )

        # The same condition is held out across all speeds.
        if set(train["condition"]) & set(test["condition"]):
            raise ValueError(
                "Condition leakage across speeds"
            )

        if (
            train["record_id"].nunique() != 24
            or test["record_id"].nunique() != 12
        ):
            raise ValueError(
                "Expected 24 training and 12 test recordings"
            )

        model = create_model()

        model.fit(
            train[FEATURE_COLUMNS],
            train["label"],
        )

        # Positive scores correspond to Faulty = 1.
        scores = model.decision_function(
            test[FEATURE_COLUMNS]
        )

        predicted_windows = (
            np.asarray(scores) >= 0
        ).astype(int)

        fold_windows = test[
            [
                "speed",
                "condition",
                "record_id",
                "label",
                "fold",
                "window_index",
            ]
        ].copy()

        fold_windows["decision_score"] = scores
        fold_windows["prediction"] = predicted_windows

        window_results.append(fold_windows)

        # Recording-level prediction:
        # average decision scores of its 30 windows.
        for record_id, group in fold_windows.groupby(
            "record_id"
        ):

            mean_score = float(
                group["decision_score"].mean()
            )

            prediction = int(mean_score >= 0)

            record_results.append({
                "record_id": record_id,
                "speed": int(group["speed"].iloc[0]),
                "condition": group["condition"].iloc[0],
                "fold": fold,
                "label": int(group["label"].iloc[0]),
                "mean_decision_score": mean_score,
                "prediction": prediction,
                "correct": int(
                    prediction == int(group["label"].iloc[0])
                ),
            })

        print(
            f"Fold {fold}: "
            f"{train['record_id'].nunique()} train records, "
            f"{test['record_id'].nunique()} test records"
        )

    windows = pd.concat(
        window_results,
        ignore_index=True
    )

    records = pd.DataFrame(record_results)

    if len(windows) != 1080 or len(records) != 36:
        raise ValueError(
            "Incomplete cross-validation predictions"
        )

    return windows, records


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(data, level, speed_label):

    true = data["label"].to_numpy(dtype=int)
    predicted = data["prediction"].to_numpy(dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        true,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        "level": level,
        "speed": speed_label,
        "total": len(data),
        "correct": int(np.sum(true == predicted)),
        "accuracy": accuracy_score(true, predicted),
        "balanced_accuracy": balanced_accuracy_score(
            true, predicted
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def create_performance_table(windows, records):

    rows = []

    for level, data in (
        ("record", records),
        ("window", windows),
    ):

        for speed in SPEEDS:

            subset = data[data["speed"] == speed]

            rows.append(
                calculate_metrics(
                    subset,
                    level,
                    f"SP{speed}"
                )
            )

        rows.append(
            calculate_metrics(
                data,
                level,
                "ALL_SPEEDS"
            )
        )

    return pd.DataFrame(rows)


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(records):

    matrix = confusion_matrix(
        records["label"],
        records["prediction"],
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Healthy", "Faulty"],
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )

    ax.set_title(
        "Gaussian SVM — Recording-Level Confusion Matrix\n"
        "SP1 + SP2 + SP3 | 3-Fold Cross-Validation",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    fig.tight_layout()

    path = RESULTS_DIR / "confusion_matrix.png"

    fig.savefig(
        path,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

def train_final_model(df):

    # Fit one model using all three speeds and
    # all 36 recordings for future use.
    model = create_model()

    model.fit(
        df[FEATURE_COLUMNS],
        df["label"],
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "speeds": SPEEDS,
            "window_seconds": 2.0,
            "record_aggregation": "mean_decision_score",
        },
        MODEL_FILE,
    )

    print(f"Saved: {MODEL_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("DRONE PROPELLER FAULT DETECTION")
    print("GAUSSIAN SVM — ALL SPEEDS")
    print("=" * 65)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    print(
        f"Loaded: {len(df)} windows, "
        f"{df['record_id'].nunique()} recordings"
    )

    windows, records = evaluate_cross_validation(df)

    performance = create_performance_table(
        windows,
        records,
    )

    performance.to_csv(
        RESULTS_DIR / "model_performance.csv",
        index=False,
    )

    records.to_csv(
        RESULTS_DIR / "record_predictions.csv",
        index=False,
    )

    save_confusion_matrix(records)

    train_final_model(df)

    print("\n=== PERFORMANCE SUMMARY ===")

    print(
        performance.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    record_correct = int(
        records["correct"].sum()
    )

    window_correct = int(
        (
            windows["label"] == windows["prediction"]
        ).sum()
    )

    print(
        f"\nRecord accuracy: {record_correct}/36"
    )

    print(
        f"Window accuracy: {window_correct}/1080"
    )

    print("\nModel training and evaluation complete.")


if __name__ == "__main__":
    main()
