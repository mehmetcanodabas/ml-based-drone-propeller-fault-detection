
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT_FILE = ROOT / "docs" / "results" / "all_window_features.csv"
OUTPUT_DIR = ROOT / "docs" / "results"

CHANNELS = ("C1", "C2")
SPEEDS = (1, 2, 3)

SELECTED = (
    "crest_factor",
    "kurtosis",
    "low_peak_frequency_hz",
    "autocorrelation_peak",
)

REJECTED = (
    "rms",
    "peak_to_peak",
    "dominant_frequency_hz",
    "bandpower_5_100",
    "bandpower_100_500",
    "bandpower_500_1500",
)

FEATURES = SELECTED + REJECTED

COLORS = {
    "healthy": "#21867A",
    "faulty": "#DC8741",
    "selected": "#2478B7",
    "rejected": "#A1A1A1",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 240,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# LOAD DATA
# ============================================================

def load_features():

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Feature CSV not found:\n{INPUT_FILE}\n"
            "Run extract_features.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "speed",
        "condition",
        "record_id",
        "label",
        "window_index",
    }

    required |= {
        f"{channel}__{feature}"
        for channel in CHANNELS
        for feature in FEATURES
    }

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
            "Expected 36 distinct recordings"
        )

    if set(df["speed"]) != set(SPEEDS):
        raise ValueError(
            "Expected SP1, SP2 and SP3"
        )

    if not df.groupby("record_id").size().eq(30).all():
        raise ValueError(
            "Expected 30 windows per recording"
        )

    feature_columns = [
        f"{channel}__{feature}"
        for channel in CHANNELS
        for feature in FEATURES
    ]

    values = df[feature_columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Feature values contain NaN or infinity"
        )

    return df


def record_medians(df):

    columns = [
        f"{channel}__{feature}"
        for channel in CHANNELS
        for feature in FEATURES
    ]

    return (
        df.groupby(
            ["speed", "record_id", "condition", "label"],
            as_index=False
        )[columns]
        .median()
    )


# ============================================================
# FIGURE 1: RMS VS LOW PEAK FREQUENCY
# ============================================================

def plot_rms_vs_low_peak(records):

    comparisons = (
        ("rms", "RMS"),
        (
            "low_peak_frequency_hz",
            "Low Peak Frequency (Hz)"
        ),
    )

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(14, 13),
        constrained_layout=True
    )

    rng = np.random.default_rng(42)

    for feature_index, (feature, label) in enumerate(
        comparisons
    ):

        for channel_index, channel in enumerate(CHANNELS):

            row = feature_index * 2 + channel_index
            column = f"{channel}__{feature}"

            for speed_index, speed in enumerate(SPEEDS):

                ax = axes[row, speed_index]

                subset = records[
                    records["speed"] == speed
                ]

                for class_value, x_center, color in (
                    (0, 0, COLORS["healthy"]),
                    (1, 1, COLORS["faulty"]),
                ):

                    group = subset[
                        subset["label"] == class_value
                    ]

                    values = group[column].to_numpy(
                        dtype=float
                    )

                    jitter = rng.uniform(
                        -0.16,
                        0.16,
                        len(values)
                    )

                    ax.scatter(
                        x_center + jitter,
                        values,
                        s=65,
                        color=color,
                        alpha=0.85,
                        edgecolors="white",
                        linewidths=0.7,
                        zorder=3,
                    )

                    ax.hlines(
                        np.median(values),
                        x_center - 0.22,
                        x_center + 0.22,
                        color="black",
                        linewidth=1.6,
                        zorder=4,
                    )

                ax.set_xlim(-0.5, 1.5)
                ax.set_xticks([0, 1])
                ax.set_xticklabels([
                    "Healthy",
                    "Faulty"
                ])

                ax.set_title(
                    f"SP{speed} | {channel} | {label}",
                    fontsize=10
                )

                ax.set_ylabel(label)
                ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "RMS vs Low Peak Frequency — Recording Medians\n"
        "Each point represents one physical recording",
        fontsize=15,
        fontweight="bold",
    )

    path = OUTPUT_DIR / "rms_vs_low_peak.png"

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# FIGURE 2: FEATURE CORRELATION HEATMAP
# ============================================================

def plot_feature_correlation(records):

    shown_features = (
        "rms",
        "peak_to_peak",
        "dominant_frequency_hz",
        "crest_factor",
        "kurtosis",
        "low_peak_frequency_hz",
        "autocorrelation_peak",
    )

    columns = [
        f"{channel}__{feature}"
        for channel in CHANNELS
        for feature in shown_features
    ]

    normalized = records[
        ["speed"] + columns
    ].copy()

    # Standardize within each speed
    for speed in SPEEDS:

        mask = normalized["speed"] == speed

        for column in columns:

            values = normalized.loc[
                mask,
                column
            ]

            mean = values.mean()
            std = values.std(ddof=0)

            if std > 0:
                normalized.loc[mask, column] = (
                    (values - mean) / std
                )
            else:
                normalized.loc[mask, column] = 0.0

    corr = (
        normalized[columns]
        .corr()
        .fillna(0)
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    image = ax.imshow(
        corr.to_numpy(),
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )

    labels = [
        column.replace("__", " / ")
        for column in columns
    ]

    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))

    ax.set_xticklabels(
        labels,
        rotation=60,
        ha="right",
        fontsize=8
    )

    ax.set_yticklabels(
        labels,
        fontsize=8
    )

    ax.set_title(
        "Feature Correlations — Within-Speed Standardized\n"
        "Recording-Level Medians",
        fontweight="bold",
        pad=18,
    )

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.035,
        pad=0.03,
        label="Pearson correlation",
    )

    fig.tight_layout()

    path = OUTPUT_DIR / "feature_correlation.png"

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# FEATURE QUALITY METRICS
# ============================================================

def calculate_feature_quality(records):

    rows = []

    for channel in CHANNELS:

        for feature in FEATURES:

            column = f"{channel}__{feature}"

            speed_spreads = []
            speed_separations = []

            for speed in SPEEDS:

                subset = records[
                    records["speed"] == speed
                ]

                healthy = subset.loc[
                    subset["label"] == 0,
                    column
                ].to_numpy(dtype=float)

                faulty = subset.loc[
                    subset["label"] == 1,
                    column
                ].to_numpy(dtype=float)

                all_values = subset[
                    column
                ].to_numpy(dtype=float)

                q25, q75 = np.percentile(
                    all_values,
                    [25, 75]
                )

                iqr = q75 - q25

                healthy_range = (
                    np.max(healthy)
                    - np.min(healthy)
                )

                if iqr > 1e-12:

                    normalized_spread = (
                        healthy_range / iqr
                    )

                else:

                    normalized_spread = (
                        0.0
                        if healthy_range <= 1e-12
                        else np.nan
                    )

                outside = (
                    (faulty < np.min(healthy))
                    |
                    (faulty > np.max(healthy))
                )

                speed_spreads.append(
                    normalized_spread
                )

                speed_separations.append(
                    float(np.mean(outside))
                )

            finite_spreads = np.asarray(
                speed_spreads,
                dtype=float
            )

            if np.isfinite(finite_spreads).any():

                healthy_variability = float(
                    np.nanmean(finite_spreads)
                )

            else:

                healthy_variability = np.nan

            rows.append({
                "feature": feature,
                "channel": channel,
                "selected": feature in SELECTED,
                "healthy_variability": healthy_variability,
                "fault_outside_healthy_fraction": float(
                    np.mean(speed_separations)
                ),
            })

    return pd.DataFrame(rows)


# ============================================================
# FIGURE 3: FEATURE QUALITY
# LABELS IN A FIXED LEFT COLUMN
# ============================================================


def plot_feature_quality(records):

    quality = calculate_feature_quality(records)

    fig, ax = plt.subplots(figsize=(17, 10))

    valid = quality[
        np.isfinite(quality["healthy_variability"])
    ].copy()

    for _, item in valid.iterrows():

        selected = bool(item["selected"])

        color = (
            COLORS["selected"]
            if selected
            else COLORS["rejected"]
        )

        marker = "o" if item["channel"] == "C1" else "s"

        x = float(item["healthy_variability"])
        y = float(item["fault_outside_healthy_fraction"])

        # Plot the feature point
        ax.scatter(
            x,
            y,
            color=color,
            marker=marker,
            s=75,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )

        # Label immediately to the LEFT of its point.
        # No arrows or connector lines.
        ax.annotate(
            f"{item['channel']} {item['feature']}",
            xy=(x, y),
            xytext=(-7, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=6.5,
            color=color,
            annotation_clip=False,
            zorder=4,
        )

    ax.set_xlabel(
        "Healthy Variability\n"
        "(Healthy Range / All-Record IQR; lower is better)"
    )

    ax.set_ylabel(
        "Faulty Recordings Outside Healthy Range\n"
        "(Fraction; higher is better)"
    )

    ax.set_ylim(-0.07, 1.10)

    # Extra space to the left for labels near X = 0
    ax.margins(x=0.22)

    ax.grid(alpha=0.2, zorder=0)

    legend = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["selected"],
            markeredgecolor="none",
            markersize=9,
            label="Selected"
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["rejected"],
            markeredgecolor="none",
            markersize=9,
            label="Rejected candidate"
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor="black",
            markersize=8,
            label="C1"
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="none",
            markerfacecolor="black",
            markersize=8,
            label="C2"
        ),
    ]

    ax.legend(handles=legend, loc="upper right")

    ax.set_title(
        "Feature Quality — Selected vs Rejected Features\n"
        "Within-Speed Comparison Using Recording Medians",
        fontweight="bold",
        fontsize=14,
        pad=18,
    )

    fig.tight_layout()

    path = OUTPUT_DIR / "feature_quality.png"

    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0.3
    )

    plt.close(fig)

    print(f"Saved: {path}")

    quality_path = OUTPUT_DIR / "feature_quality_metrics.csv"

    quality.to_csv(quality_path, index=False)

    print(f"Saved: {quality_path}")


# ============================================================
# FEATURE JUSTIFICATION TABLE
# ============================================================

def save_justification_table():

    data = [
        (
            "rms",
            "Rejected",
            "Amplitude-sensitive and strongly correlated "
            "with peak-to-peak. Removing it alone did not "
            "resolve the SP1 healthy false alarm."
        ),
        (
            "peak_to_peak",
            "Rejected",
            "Amplitude-sensitive and strongly correlated "
            "with RMS."
        ),
        (
            "dominant_frequency_hz",
            "Rejected",
            "Full-spectrum harmonic switching caused "
            "misleading differences among SP1 healthy repeats."
        ),
        (
            "bandpower_5_100",
            "Rejected",
            "Not retained after exploratory comparisons."
        ),
        (
            "bandpower_100_500",
            "Rejected",
            "Not retained after exploratory comparisons."
        ),
        (
            "bandpower_500_1500",
            "Rejected",
            "Not retained after exploratory comparisons."
        ),
        (
            "crest_factor",
            "Selected",
            "Represents peak amplitude relative to RMS."
        ),
        (
            "kurtosis",
            "Selected",
            "Characterizes the distribution and impulsiveness "
            "of vibration amplitudes."
        ),
        (
            "low_peak_frequency_hz",
            "Selected",
            "Captures spectral peaks within speed-specific "
            "frequency ranges."
        ),
        (
            "autocorrelation_peak",
            "Selected",
            "Represents periodic structure in vibration "
            "signals and is included in the final model."
        ),
    ]

    table = pd.DataFrame(
        data,
        columns=[
            "feature",
            "decision",
            "justification"
        ]
    )

    path = OUTPUT_DIR / "feature_justification.csv"

    table.to_csv(
        path,
        index=False
    )

    print(f"Saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("FEATURE SELECTION VISUALIZATION")
    print("=" * 65)

    df = load_features()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    records = record_medians(df)

    print(
        f"Loaded {len(df)} windows "
        f"from {len(records)} physical recordings."
    )

    plot_rms_vs_low_peak(records)

    plot_feature_correlation(records)

    plot_feature_quality(records)

    save_justification_table()

    print("\nVisualization complete.")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
