import os

import cv2
import numpy as np
import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_files(folder):
    return sorted(
        f
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow("image", image / 255.0)
    cv2.waitKey(1)


def image_stream(
    imagedir,
    calib,
    stride,
    camera_model,
    filename_is_timestamp=True,
    target_width=None,
    target_height=None,
):
    """Image generator for DROID-SLAM tracking."""

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    image_list = list_image_files(imagedir)[::stride]

    for frame_idx, imfile in enumerate(image_list):
        if filename_is_timestamp:
            t = float(os.path.splitext(imfile)[0]) * 1e-9
        else:
            t = float(frame_idx)

        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            if camera_model == "fisheye":
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K,
                    calib[4:],
                    np.eye(3),
                    K,
                    image.shape[:2][::-1],
                    cv2.CV_32F,
                )
                image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
            else:
                image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        if target_width is not None and target_height is not None:
            w1 = int(target_width)
            h1 = int(target_height)
            image = cv2.resize(image, (w1, h1))
        else:
            h1, w1 = h0, w0

        h2 = h1 - (h1 % 8)
        w2 = w1 - (w1 % 8)
        image = image[:h2, :w2]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= w2 / w0
        intrinsics[1::2] *= h2 / h0

        yield t, image[None], intrinsics
