import argparse
import copy
import os

import torch
import yaml
from tqdm import tqdm

from droid import Droid
from droid_async import DroidAsync

from .config import load_runs_config
from .exporters import (
    export_ply,
    export_ply_from_reconstruction_file,
    export_poses_csv,
    save_reconstruction,
)
from .io_stream import image_stream, list_image_files, show_image


def _resolve_run_paths(args):
    if args.root_folder:
        if args.input_folder and not os.path.isabs(args.input_folder):
            args.input_folder = os.path.join(args.root_folder, args.input_folder)
        if args.output_folder and not os.path.isabs(args.output_folder):
            args.output_folder = os.path.join(args.root_folder, args.output_folder)
        if args.calib and not os.path.isabs(args.calib):
            args.calib = os.path.join(args.root_folder, args.calib)


def run_tracking(args, run_name=None):
    if not args.input_folder:
        raise ValueError("missing required input folder (--input-folder)")
    if not args.calib:
        raise ValueError("missing required calibration file (--calib)")
    if (args.target_width is None) != (args.target_height is None):
        raise ValueError("--target-width and --target-height must be provided together")
    if args.target_width is not None and (args.target_width <= 0 or args.target_height <= 0):
        raise ValueError("--target-width and --target-height must be positive")

    _resolve_run_paths(args)

    args.stereo = False
    try:
        torch.multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass

    if args.asynchronous:
        args.disable_vis = True

    droid = None

    if args.output_folder is not None:
        args.upsample = True

    all_image_files = list_image_files(args.input_folder)[::args.stride]
    total_images = len(all_image_files)

    if run_name:
        print(f"\n===== Run: {run_name} =====")
    print(f"\n📸 Input:  {args.input_folder}  ({total_images} images, stride={args.stride})")
    print(f"📁 Output: {args.output_folder}")
    print(
        f"🎛️  Buffer: {args.buffer} keyframes | "
        f"filter_thresh={args.filter_thresh} | keyframe_thresh={args.keyframe_thresh}"
    )
    if args.target_width is None:
        print("🖼️  Resize: native input resolution (cropped to multiples of 8)")
    else:
        print(f"🖼️  Resize: {args.target_width}x{args.target_height} (cropped to multiples of 8)")
    print(f"⚙️  Mode:   {'async' if args.asynchronous else 'sync'} | upsample={args.upsample}")
    print()

    all_tstamps = []
    frame_count = 0
    stream_args = (
        args.input_folder,
        args.calib,
        args.stride,
        args.camera_model,
        args.filename_is_timestamp,
        args.target_width,
        args.target_height,
    )

    for t, image, intrinsics in tqdm(
        image_stream(*stream_args),
        desc="DROID-SLAM tracking",
        total=total_images,
        unit="frame",
        dynamic_ncols=True,
    ):
        if t < args.t0:
            continue

        all_tstamps.append(t)
        frame_count += 1

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = DroidAsync(args) if args.asynchronous else Droid(args)

        droid.track(t, image, intrinsics=intrinsics)

    print(f"🎬 Tracked {frame_count} frames → {droid.video.counter.value} keyframes retained")

    traj_est = droid.terminate(image_stream(*stream_args))

    if args.output_folder is not None:
        os.makedirs(args.output_folder, exist_ok=True)
        config_path = os.path.join(args.output_folder, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(vars(args), f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"📝 Config saved to {config_path}")
        print(f"Saving {frame_count} frames to {args.output_folder}")
        save_reconstruction(
            droid,
            os.path.join(args.output_folder, "reconstruction.pt"),
            poses_all=traj_est,
            tstamps_all=all_tstamps,
        )
        export_poses_csv(os.path.join(args.output_folder, "poses.csv"), traj_est, all_tstamps)
        export_ply(droid, os.path.join(args.output_folder, "reconstruction.ply"))
        print(f"🎉 Done! Results saved to {args.output_folder}")


def run_convert_mode(convert_args):
    export_ply_from_reconstruction_file(
        convert_args.input_file,
        convert_args.output_ply,
        filter_thresh=convert_args.filter_threshold,
        filter_count=convert_args.filter_count,
        min_disp_ratio=convert_args.min_disp_ratio,
    )
    print("🎉 Convert complete")


def run_batch_mode(args):
    defaults, runs = load_runs_config(args.runs_config)
    print(f"📚 Loaded batch config: {args.runs_config} ({len(runs)} runs)")

    failures = 0
    skipped = 0
    succeeded = 0
    for idx, run in enumerate(runs, start=1):
        run_name = run.get("name", f"run_{idx:02d}")
        run_args_dict = copy.deepcopy(vars(args))
        run_args_dict.update(defaults)
        run_args_dict.update(run)

        if bool(run_args_dict.get("skip", False)):
            skipped += 1
            print(f"⏭️  Skipping run: {run_name}")
            continue

        run_args = argparse.Namespace(**run_args_dict)

        try:
            run_tracking(run_args, run_name=run_name)
            succeeded += 1
        except Exception as exc:
            failures += 1
            print(f"❌ Run failed: {run_name} ({exc})")
            if not args.continue_on_error:
                raise

    print(f"\n📊 Batch summary: {succeeded}/{len(runs)} succeeded, {skipped} skipped, {failures} failed")
    if failures > 0:
        raise SystemExit(1)
