# python ./droid-slam.py \
#     --root-folder .data/tumvi/dataset-corridor1_512_16 \
#     --input-folder mav0/cam0/data \
#     --output-folder droid-slam \
#     --calib calib/tumvi.txt \
#     --buffer 2048 \
#     --disable_vis \
#     --filter_thresh 2.0 \
#     --keyframe_thresh 1.5 \
#     --camera-model fisheye

# python ./droid-slam.py \
#     --root-folder .data/hilti/2022/exp21_outside_building \
#     --input-folder alphasense/cam0/image_raw \
#     --output-folder droid-slam \
#     --calib calib/hilti.txt \
#     --buffer 2048 \
#     --disable_vis \
#     --filter_thresh 2.0 \
#     --keyframe_thresh 1.5 \
#     --camera-model fisheye

# python ./droid-slam.py \
#     --root-folder .data/reaper/2026_02_05/rosbags/opsys_test_2026_02_05_143722 \
#     --input-folder camera/image \
#     --output-folder droid-slam \
#     --calib calib/reaper.txt \
#     --buffer 2048 \
#     --disable_vis \
#     --filter_thresh 2.0 \
#     --keyframe_thresh 1.5 \
#     --camera-model radtan

python ./droid-slam.py \
    --root-folder .data/reaper/2026_02_05/rosbags/opsys_test_2026_02_05_144239 \
    --input-folder camera/image \
    --output-folder droid-slam \
    --calib calib/reaper.txt \
    --buffer 2048 \
    --disable_vis \
    --filter_thresh 2.0 \
    --keyframe_thresh 1.5 \
    --camera-model radtan

python ./droid-slam.py \
    --root-folder .data/reaper/2026_02_05/rosbags/opsys_test_2026_02_05_144901 \
    --input-folder camera/image \
    --output-folder droid-slam \
    --calib calib/reaper.txt \
    --buffer 2048 \
    --disable_vis \
    --filter_thresh 2.0 \
    --keyframe_thresh 1.5 \
    --camera-model radtan