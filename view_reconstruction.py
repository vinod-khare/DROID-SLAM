import sys
sys.path.append("droid_slam")

import torch
import argparse
import os
from datetime import datetime
import time

import droid_backends
import argparse
import open3d as o3d
from tkinter import colorchooser, filedialog
import tkinter as tk
import numpy as np

from visualization import create_camera_actor
from lietorch import SE3

from cuda_timer import CudaTimer

def parse_color(color_str):
    """Parse color string in format 'R,G,B' (values 0-255) or hex '#RRGGBB'"""
    if color_str.startswith('#'):
        # Parse hex color
        hex_str = color_str.lstrip('#')
        return [int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    else:
        # Parse RGB format (0-255)
        try:
            rgb = [int(x.strip()) for x in color_str.split(',')]
            if len(rgb) != 3 or any(c < 0 or c > 255 for c in rgb):
                raise ValueError
            return [c / 255.0 for c in rgb]
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid color format: {color_str}. Use 'R,G,B' (0-255) or '#RRGGBB'")

def get_background_color(bg_color_arg):
    """Get background color from argument or GUI color picker"""
    if bg_color_arg:
        # User provided color via command line
        return parse_color(bg_color_arg)
    else:
        # Show color picker GUI
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        root.attributes('-topmost', True)
        
        color = colorchooser.askcolor(color=(255, 255, 255), title="Select Background Color")
        root.destroy()
        
        if color[0] is None:
            # User cancelled, use default white
            return [1.0, 1.0, 1.0]
        
        # askcolor returns ((R, G, B), hex_str) with RGB in 0-255 range
        rgb = color[0]
        return [c / 255.0 for c in rgb]

class ControlWindow:
    """Tkinter control window for Open3D visualization"""
    def __init__(self, vis, root):
        self.vis = vis
        self.root = root
        self.root.title("Reconstruction Controls")
        self.root.geometry("300x150")
        
        # Background color button
        self.bg_button = tk.Button(
            root,
            text="Change Background Color",
            command=self.change_background_color,
            height=2,
            font=("Arial", 10)
        )
        self.bg_button.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Save image button
        self.save_button = tk.Button(
            root,
            text="Save View as Image",
            command=self.save_image,
            height=2,
            font=("Arial", 10)
        )
        self.save_button.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_label = tk.Label(
            root,
            text="",
            font=("Arial", 8),
            fg="green"
        )
        self.status_label.pack(pady=5)
    
    def change_background_color(self):
        """Open color picker and update background"""
        self.root.attributes('-topmost', True)
        color = colorchooser.askcolor(color=(255, 255, 255), title="Select Background Color")
        
        if color[0] is not None:
            # Convert RGB (0-255) to normalized values (0-1)
            rgb = color[0]
            normalized_color = [c / 255.0 for c in rgb]
            self.vis.get_render_option().background_color = normalized_color
            self.show_status("Background color updated")
    
    def save_image(self):
        """Save the current Open3D view as an image"""
        self.root.attributes('-topmost', True)
        # Open file dialog to select save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=f"reconstruction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        
        if filename:
            try:
                self.show_status("Rendering and saving...")
                # Force multiple render cycles to ensure the scene is properly rendered
                for _ in range(5):
                    self.vis.poll_events()
                    self.vis.update_renderer()
                    time.sleep(0.05)
                
                # Capture the screen and save
                self.vis.capture_screen_image(filename)
                
                # Verify the file was created and is not empty
                if os.path.exists(filename) and os.path.getsize(filename) > 0:
                    self.show_status(f"Image saved: {os.path.basename(filename)}")
                else:
                    self.show_status("Error: Failed to save image", error=True)
            except Exception as e:
                self.show_status(f"Error saving image: {str(e)}", error=True)
    
    def show_status(self, message, error=False):
        """Display status message"""
        self.status_label.config(text=message, fg="red" if error else "green")
        # Clear message after 3 seconds
        self.root.after(3000, lambda: self.status_label.config(text=""))

def view_reconstruction(filename: str, filter_thresh = 0.005, filter_count=2, bg_color=None):
    reconstruction_blob = torch.load(filename, weights_only=False)
    images = reconstruction_blob["images"].cuda()[...,::2,::2]
    disps = reconstruction_blob["disps"].cuda()[...,::2,::2]
    poses = reconstruction_blob["poses"].cuda()
    intrinsics = 4 * reconstruction_blob["intrinsics"].cuda()

    disps = disps.contiguous()

    index = torch.arange(len(images), device="cuda")
    thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))

    with CudaTimer("iproj"):
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
    colors = images[:,[2,1,0]].permute(0,2,3,1) / 255.0

    with CudaTimer("filter"):
        counts = droid_backends.depth_filter(poses, disps, intrinsics[0], index, thresh)

    mask = (counts >= filter_count) & (disps > .25 * disps.mean())
    points_np = points[mask].cpu().numpy()
    colors_np = colors[mask].cpu().numpy()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_np)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_np)

    vis = o3d.visualization.Visualizer()
    vis.create_window(height=960, width=960)
    vis.get_render_option().load_from_json("misc/renderoption.json")

    # Set background color from command line or default
    if bg_color:
        background_color = parse_color(bg_color)
    else:
        background_color = [1.0, 1.0, 1.0]  # Default white
    vis.get_render_option().background_color = background_color

    vis.add_geometry(point_cloud)

    # get pose matrices as a nx4x4 numpy array
    pose_mats = SE3(poses).inv().matrix().cpu().numpy()

    ### add camera actor ###
    for i in range(len(poses)):
        cam_actor = create_camera_actor(False)
        cam_actor.transform(pose_mats[i])
        vis.add_geometry(cam_actor)

    # Create control window
    control_root = tk.Tk()
    control_window = ControlWindow(vis, control_root)
    
    # Run visualization with control window
    # We need to handle both the Open3D visualizer and tkinter event loops
    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()
        
        try:
            control_root.update()
        except tk.TclError:
            # Control window was closed
            break
    
    vis.destroy_window()
    try:
        control_root.destroy()
    except:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="path to image directory")
    parser.add_argument("--filter_threshold", type=float, default=0.005)
    parser.add_argument("--filter_count", type=int, default=3)
    parser.add_argument("--bg_color", type=str, default=None,
                        help="Background color in format 'R,G,B' (0-255) or '#RRGGBB'. If not specified, shows color picker.")
    args = parser.parse_args()

    view_reconstruction(args.filename, args.filter_threshold, args.filter_count, args.bg_color)