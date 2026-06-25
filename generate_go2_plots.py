from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from pandas.errors import EmptyDataError
import shutil


LOGGER = logging.getLogger(__name__)

LEG_NAMES = {
    "FR": "Front Right",
    "FL": "Front Left",
    "RR": "Rear Right",
    "RL": "Rear Left",
}

JOINT_NAMES = {
    "hip": "Hip",
    "thigh": "Thigh",
    "calf": "Calf",
}

METRIC_NAMES = {
    "q": ("Position", "rad"),
    "dq": ("Velocity", "rad/s"),
    "tau": ("Torque", "N·m"),
}

FOOT_COLUMNS = {
    "FL": "foot_FL",
    "FR": "foot_FR",
    "RL": "foot_RL",
    "RR": "foot_RR",
}

COLOR_CYCLE = {
    "hip": "#005f73",
    "thigh": "#bb3e03",
    "calf": "#0a9396",
}


def leg_slug(leg_prefix: str) -> str:
    return LEG_NAMES[leg_prefix].lower().replace(" ", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate clear Go2 low-state plots from a CSV log. The x-axis is elapsed "
            "seconds from the first timestamp."
        )
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=Path("data/go2_lowstate_stops.csv"),
        type=Path,
        help="Path to the CSV file. Defaults to data/go2_lowstate_walking.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots will be written. Defaults to plots/<csv-stem>/.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        data = pd.read_csv(csv_path)
    except EmptyDataError as error:
        raise ValueError(f"CSV file is empty: {csv_path}") from error

    if data.empty:
        raise ValueError(f"CSV file contains no data rows: {csv_path}")

    if "timestamp" not in data.columns:
        raise ValueError("CSV is missing the required 'timestamp' column")

    timestamps = pd.to_numeric(data["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Timestamp column contains non-numeric values")

    elapsed_seconds = timestamps - timestamps.iloc[0]
    return data, elapsed_seconds


def validate_columns(data: pd.DataFrame) -> None:
    required_columns: list[str] = ["timestamp"]
    for leg_prefix in LEG_NAMES:
        for joint_name in JOINT_NAMES:
            for metric_name in METRIC_NAMES:
                required_columns.append(f"{leg_prefix}_{joint_name}_{metric_name}")
    required_columns.extend(FOOT_COLUMNS.values())
    required_columns.extend(["roll", "pitch", "yaw", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"])

    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing_columns))


def style_axes(ax: Axes) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.tick_params(axis="both", labelsize=11)


def format_time_axis(ax: Axes) -> None:
    ax.set_xlabel("Elapsed time (s)", fontsize=12)
    style_axes(ax)


def save_figure(figure: Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=220, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("Saved %s", output_path)


def plot_leg_metric(
    elapsed_seconds: pd.Series,
    data: pd.DataFrame,
    leg_prefix: str,
    metric_name: str,
    output_dir: Path,
) -> None:
    metric_label, metric_unit = METRIC_NAMES[metric_name]
    leg_label = LEG_NAMES[leg_prefix]

    figure, axis = plt.subplots(figsize=(15, 6))
    for joint_name, joint_label in JOINT_NAMES.items():
        column_name = f"{leg_prefix}_{joint_name}_{metric_name}"
        axis.plot(
            elapsed_seconds,
            data[column_name],
            label=f"{joint_label}",
            linewidth=2.0,
            color=COLOR_CYCLE[joint_name],
        )

    axis.set_title(f"{leg_label} Leg {metric_label}", fontsize=16, weight="bold")
    axis.set_ylabel(f"{metric_label} ({metric_unit})", fontsize=12)
    format_time_axis(axis)
    axis.legend(loc="upper right", ncols=3, frameon=True)

    file_name = f"{leg_slug(leg_prefix)}_{metric_label.lower()}_vs_time.png"
    save_figure(figure, output_dir / "legs" / leg_slug(leg_prefix) / file_name)


def plot_leg_summary(
    elapsed_seconds: pd.Series,
    data: pd.DataFrame,
    leg_prefix: str,
    output_dir: Path,
) -> None:
    leg_label = LEG_NAMES[leg_prefix]
    metric_order = ("q", "dq", "tau")

    figure, axes = plt.subplots(len(metric_order), 1, figsize=(15, 13), sharex=True)
    axes_list = list(axes)

    for axis, metric_name in zip(axes_list, metric_order, strict=True):
        metric_label, metric_unit = METRIC_NAMES[metric_name]
        for joint_name, joint_label in JOINT_NAMES.items():
            column_name = f"{leg_prefix}_{joint_name}_{metric_name}"
            axis.plot(
                elapsed_seconds,
                data[column_name],
                label=joint_label,
                linewidth=2.0,
                color=COLOR_CYCLE[joint_name],
            )

        axis.set_ylabel(f"{metric_label}\n({metric_unit})", fontsize=12)
        axis.set_title(f"{leg_label} Leg {metric_label}", fontsize=14, weight="bold")
        style_axes(axis)
        axis.legend(loc="upper right", ncols=3, frameon=True)

    format_time_axis(axes_list[-1])
    figure.suptitle(f"{leg_label} Leg Joint Signals", fontsize=18, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(figure, output_dir / "legs" / leg_slug(leg_prefix) / f"{leg_slug(leg_prefix)}_joint_summary.png")


def plot_foot_force(
    elapsed_seconds: pd.Series,
    data: pd.DataFrame,
    foot_prefix: str,
    output_dir: Path,
) -> None:
    foot_label = LEG_NAMES[foot_prefix]
    column_name = FOOT_COLUMNS[foot_prefix]

    figure, axis = plt.subplots(figsize=(14, 5))
    axis.plot(elapsed_seconds, data[column_name], color="#4a4e69", linewidth=2.2)
    axis.set_title(f"{foot_label} Foot Force", fontsize=16, weight="bold")
    axis.set_ylabel("Force / contact value", fontsize=12)
    format_time_axis(axis)

    file_name = f"{leg_slug(foot_prefix)}_foot_force_vs_time.png"
    save_figure(figure, output_dir / "foot_forces" / file_name)


def plot_three_axis_signal(
    elapsed_seconds: pd.Series,
    data: pd.DataFrame,
    columns: tuple[str, str, str],
    title: str,
    y_label: str,
    output_path: Path,
    labels: tuple[str, str, str] | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(15, 6))
    default_labels = ("X", "Y", "Z")
    colors = ("#005f73", "#ca6702", "#bb3e03")

    plot_labels = labels if labels is not None else default_labels

    for column_name, label, color in zip(columns, plot_labels, colors, strict=True):
        axis.plot(elapsed_seconds, data[column_name], label=label, linewidth=2.0, color=color)

    axis.set_title(title, fontsize=16, weight="bold")
    axis.set_ylabel(y_label, fontsize=12)
    format_time_axis(axis)
    axis.legend(loc="upper right", ncols=3, frameon=True)
    save_figure(figure, output_path)


def build_output_dir(csv_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return Path("plots") / csv_path.stem


def main() -> None:
    setup_logging()
    args = parse_args()

    csv_path = args.csv_path
    output_dir = build_output_dir(csv_path, args.output_dir)

    # Remove existing outputs for a clean run
    if output_dir.exists():
        LOGGER.info("Removing existing output directory: %s", output_dir)
        shutil.rmtree(output_dir)

    data, elapsed_seconds = load_data(csv_path)
    validate_columns(data)

    LOGGER.info("Loaded %d rows from %s", len(data), csv_path)
    LOGGER.info("Writing plots to %s", output_dir)

    for leg_prefix in LEG_NAMES:
        plot_leg_metric(elapsed_seconds, data, leg_prefix, "q", output_dir)
        plot_leg_metric(elapsed_seconds, data, leg_prefix, "dq", output_dir)
        plot_leg_metric(elapsed_seconds, data, leg_prefix, "tau", output_dir)
        plot_leg_summary(elapsed_seconds, data, leg_prefix, output_dir)

    for foot_prefix in FOOT_COLUMNS:
        plot_foot_force(elapsed_seconds, data, foot_prefix, output_dir)

    plot_three_axis_signal(
        elapsed_seconds,
        data,
        ("roll", "pitch", "yaw"),
        "Body Orientation: Roll, Pitch, and Yaw",
        "Angle (rad)",
        output_dir / "body_orientation" / "roll_pitch_yaw_vs_time.png",
        labels=("Roll (X)", "Pitch (Y)", "Yaw (Z)"),
    )
    plot_three_axis_signal(
        elapsed_seconds,
        data,
        ("gyro_x", "gyro_y", "gyro_z"),
        "Gyroscope: X, Y, and Z",
        "Angular rate",
        output_dir / "imu" / "gyro_xyz_vs_time.png",
    )
    plot_three_axis_signal(
        elapsed_seconds,
        data,
        ("acc_x", "acc_y", "acc_z"),
        "Accelerometer: X, Y, and Z",
        "Acceleration",
        output_dir / "imu" / "acc_xyz_vs_time.png",
    )

    LOGGER.info("Done")


if __name__ == "__main__":
    main()