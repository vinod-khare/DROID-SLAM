import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import csv
import argparse
import yaml

from torch.multiprocessing import Process
from droid import Droid
from droid_async import DroidAsync

import torch.nn.functional as F

import droid_backends
import open3d as o3d
from lietorch import SE3
from scipy.spatial.transform import Rotation


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib, stride, camera_model, filename_is_timestamp=False):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

    for frame_idx, imfile in enumerate(image_list):
        if filename_is_timestamp:
            t = float(os.path.splitext(imfile)[0]) * 1e-9
        else:
            t = float(frame_idx)
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            if camera_model == "fisheye":
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, calib[4:], np.eye(3), K, image.shape[:2][::-1], cv2.CV_32F)
                image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
            else:
                image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics


def save_reconstruction(droid, save_path, poses_all=None, tstamps_all=None):

    if hasattr(droid, "video2"):
        video = droid.video2
    else:
        video = droid.video

    t = video.counter.value
    save_data = {
        "tstamps": video.tstamp[:t].cpu(),
        "images": video.images[:t].cpu(),
        "disps": video.disps_up[:t].cpu(),
        "poses": video.poses[:t].cpu(),
        "intrinsics": video.intrinsics[:t].cpu()
    }

    if poses_all is not None:
        save_data["poses_all"] = torch.as_tensor(poses_all).cpu()

    if tstamps_all is not None:
        save_data["tstamps_all"] = torch.as_tensor(tstamps_all).cpu()

    torch.save(save_data, save_path)


def export_poses_csv(output_path, poses_all, tstamps_all):
    pose_mats = SE3(torch.as_tensor(poses_all)).matrix().cpu().numpy()
    timestamps = np.asarray(tstamps_all)

    n = min(len(timestamps), len(pose_mats))

    print(f"Saving poses to {output_path}")
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

        for i in range(n):
            pose_mat = pose_mats[i]
            position = pose_mat[:3, 3]
            quaternion = Rotation.from_matrix(pose_mat[:3, :3]).as_quat()
            writer.writerow([timestamps[i], *position.tolist(), *quaternion.tolist()])


def export_ply(droid, output_path, filter_thresh=0.005, filter_count=2):
    if hasattr(droid, "video2"):
        video = droid.video2
    else:
        video = droid.video

    t = video.counter.value
    images = video.images[:t].cuda()[..., ::2, ::2]
    disps = video.disps_up[:t].cuda()[..., ::2, ::2].contiguous()
    poses = video.poses[:t].cuda()
    intrinsics = 4 * video.intrinsics[:t].cuda()

    index = torch.arange(t, device="cuda")
    thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))

    points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
    colors = images[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.0
    counts = droid_backends.depth_filter(poses, disps, intrinsics[0], index, thresh)

    mask = (counts >= filter_count) & (disps > 0.25 * disps.mean())
    points_np = points[mask].cpu().numpy()
    colors_np = colors[mask].cpu().numpy()

    print(f"Creating point cloud with {len(points_np)} points")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    print(f"Saving point cloud to {output_path}")
    o3d.io.write_point_cloud(output_path, pcd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, help="path to image directory")
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
    parser.add_argument("--filename-is-timestamp", action="store_true", help="treat image filename stem as a nanosecond UNIX timestamp")
    parser.add_argument("--output-dir", type=str, default=None, help="folder to save reconstruction (.pt) and point cloud (.ply)")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)

    args = parser.parse_args()

    args.stereo = False
    try:
        torch.multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass  # method already set

    # Disable visualization in async mode to avoid CUDA initialization errors in child processes on WSL2
    if args.asynchronous:
        args.disable_vis = True

    droid = None

    # need high resolution depths for PLY export
    if args.output_dir is not None:
        args.upsample = True

    # Count total images for progress bar
    import glob as glob_module
    all_image_files = sorted(os.listdir(args.input_dir))[::args.stride]
    total_images = len(all_image_files)

    all_tstamps = []
    frame_count = 0
    for (t, image, intrinsics) in tqdm(image_stream(args.input_dir, args.calib, args.stride, args.camera_model, args.filename_is_timestamp),
                                         desc="DROID-SLAM tracking",
                                         total=total_images,
                                         unit="frame",
                                         dynamic_ncols=True):
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

    traj_est = droid.terminate(image_stream(args.input_dir, args.calib, args.stride, args.camera_model, args.filename_is_timestamp))
    
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        config_path = os.path.join(args.output_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(vars(args), f, default_flow_style=False)
        print(f"📝 Config saved to {config_path}")
        print(f"Saving {frame_count} frames to {args.output_dir}")
        save_reconstruction(droid, os.path.join(args.output_dir, "reconstruction.pt"), poses_all=traj_est, tstamps_all=all_tstamps)
        export_poses_csv(os.path.join(args.output_dir, "poses.csv"), traj_est, all_tstamps)
        export_ply(droid, os.path.join(args.output_dir, "reconstruction.ply"))
