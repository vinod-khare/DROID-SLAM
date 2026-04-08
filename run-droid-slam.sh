python ./droid-slam.py \
    --root-folder .data/tumvi/dataset-corridor1_512_16 \
    --input-folder mav0/cam0/data \
    --output-folder droid-slam \
    --calib calib/tumvi.txt \
    --buffer 2048 \
    --disable_vis \
    --filter_thresh 2.0 \
    --keyframe_thresh 1.5 \
    --camera-model fisheye