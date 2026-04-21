import argparse
import yaml


RUN_CONFIG_ALLOWED_KEYS = {
    "name",
    "skip",
    "root_folder",
    "input_folder",
    "output_folder",
    "calib",
    "t0",
    "stride",
    "weights",
    "buffer",
    "image_size",
    "disable_vis",
    "beta",
    "filter_thresh",
    "warmup",
    "keyframe_thresh",
    "frontend_thresh",
    "frontend_window",
    "frontend_radius",
    "frontend_nms",
    "backend_thresh",
    "backend_radius",
    "backend_nms",
    "upsample",
    "asynchronous",
    "frontend_device",
    "backend_device",
    "camera_model",
    "filename_is_timestamp",
    "target_width",
    "target_height",
}


def load_runs_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError("runs config must be a YAML mapping")

    defaults = config.get("defaults", {}) or {}
    runs = config.get("runs", None)

    if not isinstance(defaults, dict):
        raise ValueError("runs config field 'defaults' must be a mapping")
    if not isinstance(runs, list) or len(runs) == 0:
        raise ValueError("runs config must contain a non-empty 'runs' list")

    unknown_default_keys = set(defaults.keys()) - RUN_CONFIG_ALLOWED_KEYS
    if unknown_default_keys:
        raise ValueError(f"unknown keys in defaults: {sorted(unknown_default_keys)}")

    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"run #{idx + 1} must be a mapping")
        unknown_run_keys = set(run.keys()) - RUN_CONFIG_ALLOWED_KEYS
        if unknown_run_keys:
            raise ValueError(f"unknown keys in run #{idx + 1}: {sorted(unknown_run_keys)}")

    return defaults, runs


def build_convert_parser():
    parser = argparse.ArgumentParser(
        prog="droid-slam.py convert",
        description="Convert reconstruction .pt file to .ply point cloud",
    )
    parser.add_argument("--input-file", type=str, required=True, help="path to reconstruction .pt file")
    parser.add_argument("--output-ply", type=str, required=True, help="path to output .ply file")
    parser.add_argument("--filter-threshold", type=float, default=0.005, help="consistency threshold used for point filtering (higher -> denser)")
    parser.add_argument("--filter-count", type=int, default=1, help="minimum number of supporting views per point (lower -> denser)")
    parser.add_argument("--min-disp-ratio", type=float, default=0.1, help="minimum disparity as fraction of mean (lower -> denser, 0.0 to disable)")
    return parser


def build_main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-folder", type=str, default=None, help="root folder for relative paths (optional)")
    parser.add_argument("--input-folder", type=str, help="path to image directory (absolute or relative to --root-folder)")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=1, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=2.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")

    parser.add_argument("--camera-model", type=str, default="radtan", choices=["radtan", "fisheye"], help="Camera model: radtan or fisheye")
    parser.add_argument("--filename-is-timestamp", action=argparse.BooleanOptionalAction, default=True, help="treat image filename stem as a nanosecond UNIX timestamp (default: enabled; use --no-filename-is-timestamp to disable)")
    parser.add_argument("--target-width", type=int, default=None, help="optional target resize width before tracking; default uses native input width")
    parser.add_argument("--target-height", type=int, default=None, help="optional target resize height before tracking; default uses native input height")
    parser.add_argument("--output-folder", type=str, default=None, help="folder to save reconstruction (.pt) and point cloud (.ply) (absolute or relative to --root-folder)")
    parser.add_argument("--runs-config", type=str, default=None, help="YAML file defining multiple dataset runs")
    parser.add_argument("--continue-on-error", action="store_true", help="in batch mode, continue running other datasets after an error")
    return parser
