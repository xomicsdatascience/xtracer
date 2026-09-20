"""Measure M/M+1/M+2 evidence for DIA-NN precursors in PAMAF MS1 data.

The default run set contains one replicate each for 25, 50, and 100 ng. DIA-NN
RT values are converted from minutes to seconds, and the synthetic ion-mobility
coordinate is converted back to PAMAF arrival time with
AT = (IM - 0.5) / 0.0032.
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xtracer.mbi import MBIReader
from xtracer.utils import C13_DELTA, find_local_maximum, merge_frames


DEFAULT_DATA_DIR = PROJECT_ROOT / "data_mbi1" / "amount"
DEFAULT_REPORT = DEFAULT_DATA_DIR / "diann261" / "report.parquet"
IM_INTERCEPT = 0.5
IM_SLOPE = 0.0032

DEFAULT_RUNS = (
    ("25 ng", "2024-10-24 13.58.38-25ngHeLa_1iRT_CERamp-updated"),
    ("50 ng", "2024-10-24 19.23.10-50ngHeLa_1iRT_CERamp-updated"),
    ("100 ng", "2024-10-25 01.11.31-100ngHeLa_1iRT_CERamp-updated"),
)

REPORT_COLUMNS = [
    "Run",
    "Precursor.Id",
    "Modified.Sequence",
    "Precursor.Charge",
    "Precursor.Mz",
    "RT",
    "IM",
    "Q.Value",
    "Decoy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use one 25/50/100 ng run to measure M/M+1/M+2 peaks in "
            "three-frame-merged PAMAF MS1 spectra."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--q-value", type=float, default=0.01)
    parser.add_argument("--ppm", type=float, default=30.0)
    parser.add_argument("--at-tolerance", type=float, default=2.0)
    parser.add_argument(
        "--merge-ms1",
        type=int,
        default=3,
        help="Odd number of adjacent MS1 frames to merge.",
    )
    parser.add_argument(
        "--neighbor-points",
        type=int,
        default=5,
        help="xTracer local-maximum neighboring-point threshold.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional per-run precursor limit for a quick smoke test.",
    )
    args = parser.parse_args()
    if args.merge_ms1 < 1 or args.merge_ms1 % 2 != 1:
        parser.error("--merge-ms1 must be a positive odd integer")
    if args.ppm <= 0 or args.at_tolerance <= 0:
        parser.error("--ppm and --at-tolerance must be positive")
    return args


def nearest_positions(sorted_values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return positions of the nearest values in an ascending array."""
    right = np.searchsorted(sorted_values, targets, side="left")
    right = np.clip(right, 0, len(sorted_values) - 1)
    left = np.clip(right - 1, 0, len(sorted_values) - 1)
    choose_left = np.abs(sorted_values[left] - targets) <= np.abs(
        sorted_values[right] - targets
    )
    return np.where(choose_left, left, right)


def centered_window(length: int, center: int, width: int) -> np.ndarray:
    """Return a fixed-width, centered index window, shifted at boundaries."""
    if width > length:
        raise ValueError(f"Cannot merge {width} MS1 frames; only {length} exist")
    half = width // 2
    start = min(max(center - half, 0), length - width)
    return np.arange(start, start + width, dtype=np.int64)


def match_peak(
    ats: np.ndarray,
    mzs: np.ndarray,
    intensities: np.ndarray,
    target_at: float,
    target_mz: float,
    ppm: float,
    at_tolerance: float,
) -> tuple[bool, float, float, float]:
    """Select the most intense local maximum inside the AT/mass window."""
    ppm_errors = 1e6 * (mzs - target_mz) / target_mz
    mask = (np.abs(ppm_errors) <= ppm) & (
        np.abs(ats - target_at) <= at_tolerance
    )
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        return False, 0.0, np.nan, np.nan
    selected = candidates[np.argmax(intensities[candidates])]
    return (
        True,
        float(intensities[selected]),
        float(ppm_errors[selected]),
        float(ats[selected] - target_at),
    )


def read_report(report_path: Path, q_value: float) -> pd.DataFrame:
    report = pd.read_parquet(report_path, columns=REPORT_COLUMNS)
    report = report[
        (report["Decoy"] == 0)
        & (report["Q.Value"] <= q_value)
        & report["Precursor.Mz"].notna()
        & report["Precursor.Charge"].notna()
        & report["RT"].notna()
        & report["IM"].notna()
    ].copy()
    report = report.sort_values("Q.Value").drop_duplicates(
        ["Run", "Precursor.Id"], keep="first"
    )
    report["RT.seconds"] = report["RT"].astype(np.float64) * 60.0
    report["AT"] = (
        report["IM"].astype(np.float64) - IM_INTERCEPT
    ) / IM_SLOPE
    return report


def analyze_run(
    rows: pd.DataFrame,
    mbi_path: Path,
    merge_ms1: int,
    ppm: float,
    at_tolerance: float,
    neighbor_points: int,
) -> pd.DataFrame:
    reader = MBIReader(mbi_path, merge_ms1)
    try:
        frame_levels = np.asarray(reader.GetFrameMSLevels(), dtype=np.int64)
        frame_rts = np.asarray(reader.GetRetentionTimes(), dtype=np.float64)
        if frame_levels.size != frame_rts.size:
            raise RuntimeError(f"Frame metadata length mismatch in {mbi_path}")

        ms1_frame_ids = np.flatnonzero(frame_levels == 2)
        ms1_rts = frame_rts[ms1_frame_ids]
        center_positions = nearest_positions(
            ms1_rts, rows["RT.seconds"].to_numpy(dtype=np.float64)
        )
        rows = rows.copy().reset_index(drop=True)
        rows["_center_position"] = center_positions

        results: list[dict[str, object]] = []
        frame_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        grouped = rows.groupby("_center_position", sort=True)
        for group_number, (center_position, group) in enumerate(grouped, start=1):
            positions = centered_window(
                len(ms1_frame_ids), int(center_position), merge_ms1
            )
            selected_frame_ids = ms1_frame_ids[positions]
            selected_ids = {int(frame_id) for frame_id in selected_frame_ids}
            for frame_id in tuple(frame_cache):
                if frame_id not in selected_ids:
                    del frame_cache[frame_id]
            for frame_id in selected_frame_ids:
                frame_id = int(frame_id)
                if frame_id not in frame_cache:
                    frame_cache[frame_id] = reader.get_frame_data(frame_id)
            spectra = deque(frame_cache[int(frame_id)] for frame_id in selected_frame_ids)
            merged_at, merged_mz, merged_intensity = merge_frames(
                spectra, merge_ms1
            )
            local_indices = find_local_maximum(
                merged_at,
                merged_mz,
                merged_intensity,
                tol_at_area=at_tolerance,
                tol_ppm=ppm,
                tol_point_num=neighbor_points,
                mz_min=0.0,
                at_min=0.0,
            )
            peak_at = merged_at[local_indices]
            peak_mz = merged_mz[local_indices]
            peak_intensity = merged_intensity[local_indices]
            center_frame_id = int(ms1_frame_ids[int(center_position)])
            center_rt = float(frame_rts[center_frame_id])

            for _, row in group.iterrows():
                charge = int(row["Precursor.Charge"])
                mono_mz = float(row["Precursor.Mz"])
                target_at = float(row["AT"])
                target_mzs = (
                    mono_mz,
                    mono_mz + C13_DELTA / charge,
                    mono_mz + 2.0 * C13_DELTA / charge,
                )
                matches = [
                    match_peak(
                        peak_at,
                        peak_mz,
                        peak_intensity,
                        target_at,
                        target_mz,
                        ppm,
                        at_tolerance,
                    )
                    for target_mz in target_mzs
                ]
                found = [match[0] for match in matches]
                intensity = [match[1] for match in matches]
                results.append(
                    {
                        "Run": row["Run"],
                        "Precursor.Id": row["Precursor.Id"],
                        "Modified.Sequence": row["Modified.Sequence"],
                        "charge": charge,
                        "precursor_mz": mono_mz,
                        "q_value": float(row["Q.Value"]),
                        "rt_seconds": float(row["RT.seconds"]),
                        "target_at": target_at,
                        "center_ms1_frame": center_frame_id,
                        "center_rt_error_seconds": center_rt
                        - float(row["RT.seconds"]),
                        "merged_ms1_frames": ",".join(map(str, selected_frame_ids)),
                        "M_found": found[0],
                        "M1_found": found[1],
                        "M2_found": found[2],
                        "M_intensity": intensity[0],
                        "M1_intensity": intensity[1],
                        "M2_intensity": intensity[2],
                        "M1_over_M": (
                            intensity[1] / intensity[0]
                            if found[0] and found[1] and intensity[0] > 0
                            else np.nan
                        ),
                        "M2_over_M": (
                            intensity[2] / intensity[0]
                            if found[0] and found[2] and intensity[0] > 0
                            else np.nan
                        ),
                        "M_ppm_error": matches[0][2],
                        "M1_ppm_error": matches[1][2],
                        "M2_ppm_error": matches[2][2],
                        "M_at_error": matches[0][3],
                        "M1_at_error": matches[1][3],
                        "M2_at_error": matches[2][3],
                    }
                )
            if group_number % 100 == 0 or group_number == grouped.ngroups:
                print(
                    f"  analyzed MS1 groups: {group_number}/{grouped.ngroups}",
                    flush=True,
                )
        return pd.DataFrame(results)
    finally:
        reader.mbi.Close()


def count_percent(mask: pd.Series, denominator: int) -> str:
    count = int(mask.sum())
    percent = 100.0 * count / denominator if denominator else np.nan
    return f"{count} ({percent:.2f}%)"


def summarize(label: str, run: str, detail: pd.DataFrame) -> dict[str, object]:
    n_total = len(detail)
    m = detail["M_found"]
    m1 = detail["M1_found"]
    m2 = detail["M2_found"]
    m_count = int(m.sum())
    return {
        "Loading": label,
        "Run": run,
        "DIA-NN precursors": n_total,
        "M found / all": count_percent(m, n_total),
        "M not found / all": count_percent(~m, n_total),
        "M1 found / M found": count_percent(m & m1, m_count),
        "M2 found / M found": count_percent(m & m2, m_count),
        "M1 or M2 missing / M found": count_percent(m & ~(m1 & m2), m_count),
        "M only / M found": count_percent(m & ~m1 & ~m2, m_count),
        "M+M1 only / M found": count_percent(m & m1 & ~m2, m_count),
        "M+M2 no M1 / M found": count_percent(m & ~m1 & m2, m_count),
        "M+M1+M2 / M found": count_percent(m & m1 & m2, m_count),
        "M1/M median": detail.loc[m & m1, "M1_over_M"].median(),
        "M2/M median": detail.loc[m & m2, "M2_over_M"].median(),
        "median |RT error| s": detail["center_rt_error_seconds"].abs().median(),
    }


def main() -> int:
    args = parse_args()
    print("DIA-NN isotope analysis")
    print(f"report: {args.report.resolve()}")
    print(f"q-value <= {args.q_value}")
    print("RT conversion: seconds = DIA-NN RT minutes * 60")
    print(f"IM conversion: AT = (IM - {IM_INTERCEPT}) / {IM_SLOPE}")
    print(f"merged adjacent MS1 frames: {args.merge_ms1}")
    print(f"peak window: {args.ppm} ppm, AT +/- {args.at_tolerance} ms")
    print(f"local-maximum neighbor threshold: {args.neighbor_points}")
    print("M-1: not evaluated")

    report = read_report(args.report, args.q_value)
    summaries: list[dict[str, object]] = []
    for label, run in DEFAULT_RUNS:
        mbi_path = args.data_dir / f"{run}.mbi"
        rows = report[report["Run"] == run].copy()
        if args.limit is not None:
            rows = rows.head(args.limit)
        if rows.empty:
            raise RuntimeError(f"No passing DIA-NN precursors for run: {run}")
        if not mbi_path.is_file():
            raise FileNotFoundError(mbi_path)
        print(f"\n[{label}] {run}")
        print(f"  MBI: {mbi_path.resolve()}")
        print(f"  passing unique precursors: {len(rows)}")
        detail = analyze_run(
            rows,
            mbi_path,
            merge_ms1=args.merge_ms1,
            ppm=args.ppm,
            at_tolerance=args.at_tolerance,
            neighbor_points=args.neighbor_points,
        )
        summaries.append(summarize(label, run, detail))

    summary = pd.DataFrame(summaries)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    print("\nSummary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
