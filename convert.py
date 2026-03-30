import sys
sys.path.append("droid_slam")

import torch
import argparse
import numpy as np
import csv
import open3d as o3d
from scipy.spatial.transform import Rotation

import droid_backends
from lietorch import SE3
from cuda_timer import CudaTimer


def convert_reconstruction(filename: str, output_ply: str, output_csv: str, 
                          filter_thresh=0.005, filter_count=2):
    """
    Convert reconstruction data to PLY point cloud and CSV poses.
    
    Args:
        filename: Path to saved reconstruction file
        output_ply: Output path for PLY point cloud
        output_csv: Output path for CSV poses
        filter_thresh: Threshold for depth filtering
        filter_count: Minimum count for filtering
    """
    print(f"Loading reconstruction from {filename}")
    reconstruction_blob = torch.load(filename)
    
    images = reconstruction_blob["images"].cuda()[..., ::2, ::2]
    disps = reconstruction_blob["disps"].cuda()[..., ::2, ::2]
    poses = reconstruction_blob["poses"].cuda()
    intrinsics = 4 * reconstruction_blob["intrinsics"].cuda()
    tstamps = reconstruction_blob["tstamps"].cpu().numpy()

    poses_all = reconstruction_blob.get("poses_all", None)
    tstamps_all = reconstruction_blob.get("tstamps_all", None)

    disps = disps.contiguous()

    index = torch.arange(len(images), device="cuda")
    thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))

    print("Computing 3D points from disparities")
    with CudaTimer("iproj"):
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
    
    colors = images[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.0

    print("Filtering depth")
    with CudaTimer("filter"):
        counts = droid_backends.depth_filter(poses, disps, intrinsics[0], index, thresh)

    mask = (counts >= filter_count) & (disps > 0.25 * disps.mean())
    points_np = points[mask].cpu().numpy()
    colors_np = colors[mask].cpu().numpy()

    # Create and save point cloud
    print(f"Creating point cloud with {len(points_np)} points")
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_np)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_np)

    print(f"Saving point cloud to {output_ply}")
    o3d.io.write_point_cloud(output_ply, point_cloud)

    # Save poses as CSV (prefer all-frame trajectory if present)
    print(f"Saving poses to {output_csv}")
    if poses_all is not None:
        print("Found poses_all in reconstruction blob: exporting all-frame poses")
        poses_csv = poses_all.cuda() if isinstance(poses_all, torch.Tensor) else torch.as_tensor(poses_all, device="cuda")
        pose_mats = SE3(poses_csv).matrix().cpu().numpy()
        tstamps_csv = tstamps_all.cpu().numpy() if isinstance(tstamps_all, torch.Tensor) else np.asarray(tstamps_all)
        if tstamps_all is None:
            tstamps_csv = np.arange(len(pose_mats))
    else:
        print("poses_all not found: exporting keyframe poses")
        pose_mats = SE3(poses).inv().matrix().cpu().numpy()
        tstamps_csv = tstamps
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        
        # Write header: timestamp, position (x,y,z), quaternion (qx,qy,qz,qw)
        header = ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        writer.writerow(header)
        
        # Write poses
        n = min(len(tstamps_csv), len(pose_mats))
        for i in range(n):
            t = tstamps_csv[i]
            pose_mat = pose_mats[i]
            timestamp = np.int64(t)
            # Extract position (translation)
            position = pose_mat[:3, 3]
            
            # Extract rotation matrix and convert to quaternion
            rotation_mat = pose_mat[:3, :3]
            rotation = Rotation.from_matrix(rotation_mat)
            quaternion = rotation.as_quat()  # Returns [qx, qy, qz, qw]
            
            row = [str(timestamp)] + position.tolist() + quaternion.tolist()
            writer.writerow(row)
    
    print("Conversion complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert DROID-SLAM reconstruction to PLY and CSV")
    parser.add_argument("input", type=str, help="path to reconstruction file")
    parser.add_argument("--output_ply", type=str, default="reconstruction.ply", 
                        help="output path for PLY point cloud")
    parser.add_argument("--output_csv", type=str, default="poses.csv",
                        help="output path for CSV poses")
    parser.add_argument("--filter_threshold", type=float, default=0.005,
                        help="depth filter threshold")
    parser.add_argument("--filter_count", type=int, default=2,
                        help="minimum count for filtering")
    
    args = parser.parse_args()

    convert_reconstruction(
        args.input,
        args.output_ply,
        args.output_csv,
        args.filter_threshold,
        args.filter_count
    )
