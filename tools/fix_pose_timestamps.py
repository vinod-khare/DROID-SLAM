#!/usr/bin/env python3
"""Rewrite poses.csv timestamps from image filenames.

Assumes image filename stems are timestamps (commonly in nanoseconds) and rewrites
only the `timestamp` column while preserving pose values.
"""

import argparse
import csv
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_files(image_dir: Path, stride: int) -> list[Path]:
    files = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return files[::stride]


def extract_timestamp_from_name(path: Path, scale: float) -> float:
    try:
        return float(path.stem) * scale
    except ValueError as exc:
        raise ValueError(f"Non-numeric image filename stem: {path.name}") from exc


def read_pose_rows(csv_path: Path) -> tuple[list[dict], list[str]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        if "timestamp" not in fieldnames:
            raise ValueError("CSV must contain a 'timestamp' column")
        rows = list(reader)
    return rows, fieldnames


def write_pose_rows(csv_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix timestamps in poses.csv using image filenames")
    parser.add_argument("--poses", required=True, help="Path to input poses.csv")
    parser.add_argument("--images", required=True, help="Path to image folder")
    parser.add_argument("--output", default=None, help="Output CSV path (default: overwrite input and write .bak)")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride used when running SLAM")
    parser.add_argument("--timestamp-scale", type=float, default=1e-9, help="Scale applied to filename stem (default 1e-9 for nanoseconds to seconds)")
    args = parser.parse_args()

    poses_path = Path(args.poses)
    images_dir = Path(args.images)

    if not poses_path.exists():
        raise FileNotFoundError(f"Poses CSV not found: {poses_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {images_dir}")
    if args.stride <= 0:
        raise ValueError("--stride must be >= 1")

    image_files = list_image_files(images_dir, args.stride)
    if not image_files:
        raise ValueError(f"No image files found in: {images_dir}")

    timestamps = [extract_timestamp_from_name(p, args.timestamp_scale) for p in image_files]

    rows, fieldnames = read_pose_rows(poses_path)
    if not rows:
        raise ValueError("No pose rows found in CSV")

    n_rows = len(rows)
    n_ts = len(timestamps)
    n = min(n_rows, n_ts)

    if n_rows != n_ts:
        print(f"Warning: row/image count mismatch: poses={n_rows}, images={n_ts}. Rewriting first {n} rows.")

    for i in range(n):
        rows[i]["timestamp"] = f"{timestamps[i]:.9f}"

    if args.output:
        output_path = Path(args.output)
    else:
        backup_path = poses_path.with_suffix(poses_path.suffix + ".bak")
        backup_path.write_text(poses_path.read_text())
        output_path = poses_path
        print(f"Backup written: {backup_path}")

    write_pose_rows(output_path, rows[:n], fieldnames)

    print(f"Wrote corrected poses: {output_path}")
    print(f"Rows written: {n}")
    print(f"First timestamp: {rows[0]['timestamp']}")
    print(f"Last timestamp: {rows[n-1]['timestamp']}")


if __name__ == "__main__":
    main()
