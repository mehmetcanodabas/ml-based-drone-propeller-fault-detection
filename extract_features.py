from pathlib import Path

import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.signal import welch, correlate, find_peaks
from scipy.stats import kurtosis


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "DronePropB Ground Testing Dataset"

MODEL_OUTPUT_DIR = ROOT / "model_training"
DIAGNOSTIC_OUTPUT_DIR = ROOT / "docs" / "results"

SPEEDS = (1, 2, 3)
CHANNELS = ("C1", "C2")

WINDOW_SECONDS = 2.0
WINDOWS_PER_RECORD = 30

# Speed-specific low peak frequency bands
LOW_PEAK_BANDS = {
    1: (45.0, 70.0),
    2: (80.0, 130.0),
    3: (110.0, 145.0),
}

# Autocorrelation settings
MIN_LAG_SECONDS = 0.003
MAX_LAG_SECONDS = 0.050

# Final model: four features per channel
MODEL_FEATURE_NAMES = (
    "crest_factor",
    "kurtosis",
    "low_peak_frequency_hz",
    "autocorrelation_peak",
)

# Additional features for justification figures
DIAGNOSTIC_FEATURE_NAMES = (
    "rms",
    "peak_to_peak",
    "dominant_frequency_hz",
    "bandpower_5_100",
    "bandpower_100_500",
    "bandpower_500_1500",
)

MODEL_FEATURES = [
    f"{channel}__{feature}"
    for channel in CHANNELS
    for feature in MODEL_FEATURE_NAMES
]


# ============================================================
# DATASET CONDITIONS
# ============================================================

def build_conditions():

    conditions = []

    # Healthy recordings
    for repeat in (1, 2, 3):

        conditions.append({
            "condition": f"F0-R{repeat}",
            "label": 0,
            "fold": repeat,
            "fault": 0,
            "severity": 0,
            "repeat": repeat,
        })

    # Faulty recordings
    for fault in (1, 2, 3):

        for severity in (1, 2, 3):

            conditions.append({
                "condition": f"F{fault}-SV{severity}",
                "label": 1,
                "fold": severity,
                "fault": fault,
                "severity": severity,
                "repeat": 0,
            })

    return conditions


def make_filename(speed, condition, channel):

    fault = condition["fault"]

    if fault == 0:

        repeat = condition["repeat"]

        return (
            f"F0_SV0_SP{speed}_{channel}_R{repeat}.mat"
        )

    severity = condition["severity"]

    return (
        f"F{fault}_SV{severity}_SP{speed}_{channel}.mat"
    )


# ============================================================
# MAT FILE LOADING
# ============================================================

def read_signal(path):

    data = loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False,
    )

    if "AccData" not in data:
        raise ValueError(
            f"AccData not found: {path.name}"
        )

    acc = data["AccData"]

    signal = np.asarray(
        acc.Vib_Data,
        dtype=float
    ).reshape(-1)

    time = np.asarray(
        acc.t,
        dtype=float
    ).reshape(-1)

    if len(signal) != len(time):
        raise ValueError(
            f"Signal/time length mismatch: {path.name}"
        )

    if len(signal) < 2:
        raise ValueError(
            f"Signal too short: {path.name}"
        )

    if not np.isfinite(signal).all():
        raise ValueError(
            f"Invalid signal: {path.name}"
        )

    if not np.isfinite(time).all():
        raise ValueError(
            f"Invalid time values: {path.name}"
        )

    dt = float(np.median(np.diff(time)))

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(
            f"Invalid sampling interval: {path.name}"
        )

    fs = 1.0 / dt

    return signal, fs


def index_mat_files():

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATA_DIR}"
        )

    files = {}

    for path in DATA_DIR.rglob("*.mat"):

        if path.name in files:
            raise ValueError(
                f"Duplicate MAT file: {path.name}"
            )

        files[path.name] = path

    return files


# ============================================================
# AUTOCORRELATION PEAK
# ============================================================

def autocorrelation_peak(signal, fs):

    x = np.asarray(signal, dtype=float)

    x = x - np.mean(x)

    energy = float(np.dot(x, x))

    if energy <= 1e-15:
        return 0.0

    min_lag = max(
        1,
        int(round(MIN_LAG_SECONDS * fs))
    )

    max_lag = min(
        len(x) - 1,
        int(round(MAX_LAG_SECONDS * fs))
    )

    if min_lag >= max_lag:
        raise ValueError(
            "Invalid autocorrelation lag range"
        )

    # Positive-lag autocorrelation calculated via FFT
    ac = correlate(
        x,
        x,
        mode="full",
        method="fft"
    )[len(x) - 1:]

    ac = ac / energy

    segment = ac[min_lag:max_lag + 1]

    peaks, _ = find_peaks(segment)

    if len(peaks) == 0:
        return 0.0

    return float(
        max(0.0, np.max(segment[peaks]))
    )


# ============================================================
# FREQUENCY FEATURES
# ============================================================

def peak_frequency(freq, power, low, high):

    mask = (
        (freq >= low)
        & (freq <= high)
    )

    if not np.any(mask):
        raise ValueError(
            f"No frequency bins within {low}-{high} Hz"
        )

    selected_freq = freq[mask]
    selected_power = power[mask]

    return float(
        selected_freq[np.argmax(selected_power)]
    )


def bandpower(freq, power, low, high):

    mask = (
        (freq >= low)
        & (freq <= high)
    )

    if np.count_nonzero(mask) < 2:
        return 0.0

    return float(
        np.trapezoid(power[mask], freq[mask])
        if hasattr(np, "trapezoid")
        else np.trapz(power[mask], freq[mask])
    )


# ============================================================
# WINDOW FEATURE EXTRACTION
# ============================================================

def extract_window_features(window, fs, speed):

    x = np.asarray(window, dtype=float)

    # --------------------------------------------------------
    # Amplitude features
    # --------------------------------------------------------

    rms = float(
        np.sqrt(np.mean(x ** 2))
    )

    peak_to_peak = float(
        np.ptp(x)
    )

    peak_abs = float(
        np.max(np.abs(x))
    )

    crest_factor = (
        peak_abs / rms
        if rms > 0
        else 0.0
    )

    signal_kurtosis = float(
        kurtosis(
            x,
            fisher=False,
            bias=False
        )
    )

    # --------------------------------------------------------
    # Spectral features
    # --------------------------------------------------------

    freq, power = welch(
        x,
        fs=fs,
        nperseg=len(x)
    )

    dominant_frequency = float(
        freq[np.argmax(power)]
    )

    low, high = LOW_PEAK_BANDS[speed]

    low_peak = peak_frequency(
        freq,
        power,
        low,
        high
    )

    # --------------------------------------------------------
    # Autocorrelation feature
    # --------------------------------------------------------

    autocorr = autocorrelation_peak(
        x,
        fs
    )

    # --------------------------------------------------------
    # Final features + diagnostics
    # --------------------------------------------------------

    return {
        # Selected features
        "crest_factor": crest_factor,
        "kurtosis": signal_kurtosis,
        "low_peak_frequency_hz": low_peak,
        "autocorrelation_peak": autocorr,

        # Candidate / rejected features
        "rms": rms,
        "peak_to_peak": peak_to_peak,
        "dominant_frequency_hz": dominant_frequency,

        "bandpower_5_100": bandpower(
            freq,
            power,
            5,
            100
        ),

        "bandpower_100_500": bandpower(
            freq,
            power,
            100,
            500
        ),

        "bandpower_500_1500": bandpower(
            freq,
            power,
            500,
            1500
        ),
    }


# ============================================================
# EXTRACT ALL RECORDINGS
# ============================================================

def extract_dataset():

    file_index = index_mat_files()

    rows = []

    conditions = build_conditions()

    for speed in SPEEDS:

        for condition in conditions:

            channel_features = {}

            for channel in CHANNELS:

                filename = make_filename(
                    speed,
                    condition,
                    channel
                )

                if filename not in file_index:
                    raise FileNotFoundError(
                        f"Missing MAT file: {filename}"
                    )

                signal, fs = read_signal(
                    file_index[filename]
                )

                window_size = int(
                    round(WINDOW_SECONDS * fs)
                )

                if window_size < 2:
                    raise ValueError(
                        f"Invalid window size: {filename}"
                    )

                available_windows = (
                    len(signal) // window_size
                )

                if available_windows < WINDOWS_PER_RECORD:
                    raise ValueError(
                        f"Not enough windows: {filename}"
                    )

                extracted = []

                for i in range(WINDOWS_PER_RECORD):

                    start = i * window_size
                    end = start + window_size

                    window = signal[start:end]

                    features = extract_window_features(
                        window,
                        fs,
                        speed
                    )

                    extracted.append(features)

                channel_features[channel] = extracted

            # ------------------------------------------------
            # Merge C1 and C2 features by window index
            # ------------------------------------------------

            for i in range(WINDOWS_PER_RECORD):

                row = {
                    "speed": speed,
                    "condition": condition["condition"],
                    "record_id": (
                        f"SP{speed}_{condition['condition']}"
                    ),
                    "label": condition["label"],
                    "fold": condition["fold"],
                    "fault": condition["fault"],
                    "severity": condition["severity"],
                    "window_index": i,
                    "relative_start_s": (
                        i * WINDOW_SECONDS
                    ),
                }

                for channel in CHANNELS:

                    features = channel_features[channel][i]

                    for name, value in features.items():

                        row[
                            f"{channel}__{name}"
                        ] = value

                rows.append(row)

            print(
                f"SP{speed} {condition['condition']}: "
                f"{WINDOWS_PER_RECORD} windows"
            )

    return pd.DataFrame(rows)


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(df):

    if len(df) != 1080:
        raise ValueError(
            f"Expected 1080 rows, got {len(df)}"
        )

    if df["record_id"].nunique() != 36:
        raise ValueError(
            "Expected 36 physical recordings"
        )

    if df.duplicated(
        ["record_id", "window_index"]
    ).any():
        raise ValueError(
            "Duplicate recording/window combination"
        )

    if not df.groupby(
        "record_id"
    ).size().eq(30).all():
        raise ValueError(
            "Each recording must contain 30 windows"
        )

    if set(df["fold"]) != {1, 2, 3}:
        raise ValueError(
            "Expected folds 1, 2, 3"
        )

    if not df.groupby("record_id")[
        ["label", "fold", "speed"]
    ].nunique().eq(1).all().all():
        raise ValueError(
            "Inconsistent recording metadata"
        )

    all_feature_columns = [
        f"{channel}__{name}"
        for channel in CHANNELS
        for name in (
            MODEL_FEATURE_NAMES
            + DIAGNOSTIC_FEATURE_NAMES
        )
    ]

    if not np.isfinite(
        df[all_feature_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite feature values"
        )


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(df):

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    DIAGNOSTIC_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Final training dataset
    # --------------------------------------------------------

    metadata = [
        "speed",
        "condition",
        "record_id",
        "label",
        "fold",
        "window_index",
        "relative_start_s",
    ]

    model_df = df[
        metadata + MODEL_FEATURES
    ].copy()

    model_path = (
        MODEL_OUTPUT_DIR / "features.csv"
    )

    model_df.to_csv(
        model_path,
        index=False
    )

    # --------------------------------------------------------
    # 2. All features for justification figures
    # --------------------------------------------------------

    diagnostic_path = (
        DIAGNOSTIC_OUTPUT_DIR
        / "all_window_features.csv"
    )

    df.to_csv(
        diagnostic_path,
        index=False
    )

    print("\n=== SAVED FILES ===")

    print("\nModel training:")
    print(model_path)

    print("\nFeature analysis:")
    print(diagnostic_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("DRONE PROPELLER FAULT DETECTION")
    print("FINAL FEATURE EXTRACTION")
    print("=" * 65)

    print("Speeds: SP1 + SP2 + SP3")
    print("Sensors: C1 + C2")
    print("Window length: 2 seconds")
    print("Model features: 8")
    print("Model training output: model_training/features.csv")

    df = extract_dataset()

    validate_dataset(df)

    save_outputs(df)

    print("\n=== EXTRACTION SUMMARY ===")

    print(
        f"Physical recordings: "
        f"{df['record_id'].nunique()}"
    )

    print(
        f"Total windows: {len(df)}"
    )

    print(
        f"Model features: {len(MODEL_FEATURES)}"
    )

    print("\nWindows by speed:")

    print(
        df.groupby("speed").size().to_string()
    )

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
