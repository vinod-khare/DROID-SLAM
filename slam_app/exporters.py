import csv
import os

import droid_backends
import numpy as np
import open3d as o3d
import torch
from lietorch import SE3
from scipy.spatial.transform import Rotation


def save_reconstruction(droid, save_path, poses_all=None, tstamps_all=None):
    video = droid.video2 if hasattr(droid, "video2") else droid.video

    t = video.counter.value
    save_data = {
        "tstamps": video.tstamp[:t].cpu(),
        "images": video.images[:t].cpu(),
        "disps": video.disps_up[:t].cpu(),
        "poses": video.poses[:t].cpu(),
        "intrinsics": video.intrinsics[:t].cpu(),
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

    print(f"💾 Saving poses to {output_path}")
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"])

        for i in range(n):
            pose_mat = pose_mats[i]
            position = pose_mat[:3, 3]
            quaternion = Rotation.from_matrix(pose_mat[:3, :3]).as_quat()
            timestamp_str = f"{float(timestamps[i]):.9f}"
            writer.writerow([timestamp_str, *position.tolist(), *quaternion.tolist()])


def export_ply(droid, output_path, filter_thresh=0.005, filter_count=2):
    video = droid.video2 if hasattr(droid, "video2") else droid.video

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

    print(f"📦 Creating point cloud with {len(points_np)} points")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    print(f"💾 Saving point cloud to {output_path}")
    o3d.io.write_point_cloud(output_path, pcd)


def export_ply_from_reconstruction_file(input_file, output_ply, filter_thresh=0.005, filter_count=1, min_disp_ratio=0.1):
    """Convert a saved reconstruction .pt file into a filtered .ply point cloud."""
    print(f"📦 Loading reconstruction from {input_file}")
    reconstruction_blob = torch.load(input_file, weights_only=False)

    required_keys = ["images", "disps", "poses", "intrinsics"]
    missing = [k for k in required_keys if k not in reconstruction_blob]
    if missing:
        raise KeyError(f"Missing keys in reconstruction file: {missing}")

    images = reconstruction_blob["images"].cuda()[..., ::2, ::2]
    disps = reconstruction_blob["disps"].cuda()[..., ::2, ::2].contiguous()
    poses = reconstruction_blob["poses"].cuda()
    intrinsics = 4 * reconstruction_blob["intrinsics"].cuda()

    t = images.shape[0]
    index = torch.arange(t, device="cuda")
    thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))

    points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
    colors = images[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.0
    counts = droid_backends.depth_filter(poses, disps, intrinsics[0], index, thresh)

    mask = counts >= filter_count
    if min_disp_ratio > 0:
        mask = mask & (disps > min_disp_ratio * disps.mean())

    points_np = points[mask].cpu().numpy()
    colors_np = colors[mask].cpu().numpy()

    print(f"📦 Creating point cloud with {len(points_np)} points")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    out_dir = os.path.dirname(output_ply)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"💾 Saving point cloud to {output_ply}")
    o3d.io.write_point_cloud(output_ply, pcd)
