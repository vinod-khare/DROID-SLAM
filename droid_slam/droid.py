import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        print(f"📦 Allocating keyframe buffer (capacity: {args.buffer} frames, image size: {args.image_size}) ...")
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from visualizer.droid_visualizer import visualization_fn
            self.visualizer = Process(target=visualization_fn, args=(self.video, None))
            self.visualizer.start()
            print("🖥️  Visualizer started")
        else:
            print("🙈 Visualization disabled")

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)
        print("🚀 DROID-SLAM initialized — ready to track")


    def load_weights(self, weights):
        """ load trained model weights """

        print(f"🧠 Loading model weights from {weights} ...")
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()
        print("✅ Model weights loaded")

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """

        print("\n🏁 Tracking complete — running global optimization ...")
        del self.frontend

        import time
        torch.cuda.empty_cache()
        t = self.video.counter.value
        print(f"\n🔧 [1/2] Global bundle adjustment — {t} keyframes, 7 steps ...")
        if t > 300:
            print(f"   ⚠️  Large keyframe count ({t}) — this pass may take a while")
        t0 = time.time()
        self.backend(7)
        print(f"   ✅ Pass 1 done ({time.time() - t0:.1f}s)")

        torch.cuda.empty_cache()
        t = self.video.counter.value
        print(f"\n🔧 [2/2] Global bundle adjustment — {t} keyframes, 12 steps ...")
        if t > 300:
            print(f"   ⚠️  Large keyframe count ({t}) — this pass may take a while")
        t0 = time.time()
        self.backend(12)
        print(f"   ✅ Pass 2 done ({time.time() - t0:.1f}s)")

        print("\n📐 Filling in poses for non-keyframes ...")
        t0 = time.time()
        camera_trajectory = self.traj_filler(stream)
        print(f"   ✅ Trajectory filled ({time.time() - t0:.1f}s)")
        return camera_trajectory.inv().data.cpu().numpy()

