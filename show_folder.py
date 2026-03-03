import sys
import os
import cv2
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
import argparse
from pathlib import Path


class ImagePlayer:
    def __init__(self, root, folder_path=None):
        self.root = root
        self.root.title("Image Folder Player")
        self.root.geometry("1000x800")
        
        # State variables
        self.images = []
        self.current_frame = 0
        self.is_playing = False
        self.fps = 30
        self.folder_path = folder_path
        self.updating_slider = False  # Flag to prevent slider callback during programmatic updates
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI elements"""
        
        # Top frame for folder selection
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(top_frame, text="Load Folder", command=self.load_folder).pack(side=tk.LEFT, padx=5)
        self.folder_label = ttk.Label(top_frame, text="No folder selected", foreground="gray")
        self.folder_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Image display area
        self.image_label = ttk.Label(self.root, background="black")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Play/Pause button
        self.play_button = ttk.Button(control_frame, text="Play", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=2)
        
        # Previous/Next buttons
        ttk.Button(control_frame, text="< Prev", command=self.prev_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Next >", command=self.next_frame).pack(side=tk.LEFT, padx=2)
        
        # Frame info
        self.info_label = ttk.Label(control_frame, text="No images loaded")
        self.info_label.pack(side=tk.LEFT, padx=10)
        
        # FPS control
        ttk.Label(control_frame, text="FPS:").pack(side=tk.LEFT, padx=5)
        self.fps_var = tk.IntVar(value=30)
        fps_spinbox = ttk.Spinbox(control_frame, from_=1, to=60, textvariable=self.fps_var, width=5,
                                   command=self.update_fps)
        fps_spinbox.pack(side=tk.LEFT, padx=2)
        
        # Slider frame
        slider_frame = ttk.Frame(self.root)
        slider_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.slider = ttk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                command=self.slider_changed)
        self.slider.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)
        
        self.frame_num_label = ttk.Label(slider_frame, text="0/0", width=10)
        self.frame_num_label.pack(side=tk.LEFT, padx=5)
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Auto-load folder if provided
        if self.folder_path:
            self.root.after(100, self._load_folder_on_startup)
        
        # Start display update loop
        self.update_display()
        
    def load_folder(self):
        """Load images from folder"""
        folder = filedialog.askdirectory(title="Select folder with images")
        if not folder:
            return
        
        self._load_images_from_folder(folder)
        
    def _load_folder_on_startup(self):
        """Load folder provided at startup without showing dialog"""
        if os.path.isdir(self.folder_path):
            self._load_images_from_folder(self.folder_path)
        else:
            self.info_label.config(text=f"Error: Folder not found: {self.folder_path}")
    
    def _load_images_from_folder(self, folder):
        """Internal method to load images from a folder"""
        self.folder_path = folder
        self.folder_label.config(text=f"Folder: {folder}")
        
        # Load image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        self.images = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if Path(f).suffix.lower() in image_extensions
        ])
        
        if not self.images:
            self.info_label.config(text="No images found in folder")
            self.images = []
            return
        
        self.current_frame = 0
        self.is_playing = False
        self.play_button.config(text="Play")
        self.slider.config(to=len(self.images) - 1)
        self.info_label.config(text=f"Loaded {len(self.images)} images")
        
    def toggle_play(self):
        """Toggle play/pause"""
        if not self.images:
            return
        
        self.is_playing = not self.is_playing
        self.play_button.config(text="Pause" if self.is_playing else "Play")
        
    def next_frame(self):
        """Go to next frame"""
        if self.images:
            self.current_frame = min(self.current_frame + 1, len(self.images) - 1)
            self.slider.set(self.current_frame)
            self.is_playing = False
            self.play_button.config(text="Play")
            
    def prev_frame(self):
        """Go to previous frame"""
        if self.images:
            self.current_frame = max(self.current_frame - 1, 0)
            self.slider.set(self.current_frame)
            self.is_playing = False
            self.play_button.config(text="Play")
            
    def slider_changed(self, value):
        """Handle slider movement"""
        # Ignore callback if slider is being updated programmatically
        if self.updating_slider:
            return
        if self.images:
            self.current_frame = int(float(value))
            self.is_playing = False
            self.play_button.config(text="Play")
            
    def update_fps(self):
        """Update FPS setting"""
        self.fps = self.fps_var.get()
        
    def update_display(self):
        """Update image display"""
        if self.images:
            # Load and display current image
            img_path = self.images[self.current_frame]
            img = cv2.imread(img_path)
            
            if img is not None:
                # Resize to fit window
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                
                # Calculate scaling to fit in display area
                max_width = 1000
                max_height = 600
                scale = min(max_width / w, max_height / h, 1.0)
                
                if scale < 1.0:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Convert to PIL and display
                pil_img = Image.fromarray(img)
                tk_img = ImageTk.PhotoImage(pil_img)
                
                self.image_label.config(image=tk_img)
                self.image_label.image = tk_img  # Keep a reference
                
                # Update frame info
                self.frame_num_label.config(text=f"{self.current_frame + 1}/{len(self.images)}")
                self.updating_slider = True
                self.slider.set(self.current_frame)
                self.updating_slider = False
                
                # Handle auto-play
                if self.is_playing and self.current_frame < len(self.images) - 1:
                    self.current_frame += 1
                elif self.is_playing and self.current_frame >= len(self.images) - 1:
                    self.is_playing = False
                    self.play_button.config(text="Play")
        
        # Schedule next update
        delay = int(1000 / self.fps) if self.is_playing else 50
        self.root.after(delay, self.update_display)
        
    def on_closing(self):
        """Handle window closing"""
        self.is_playing = False
        self.root.destroy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Folder Player")
    parser.add_argument('folder', nargs='?', default=None, 
                        help="Path to folder containing images to display")
    args = parser.parse_args()
    
    root = tk.Tk()
    app = ImagePlayer(root, folder_path=args.folder)
    root.mainloop()
