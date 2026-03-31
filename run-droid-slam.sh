python ./droid-slam.py \
    --input-dir .data/reaper/2026_02_05/rosbags/opsys_test_2026_02_05_143722/camera/image \
    --output-dir .data/reaper/2026_02_05/rosbags/opsys_test_2026_02_05_143722/droid-slam-2 \
    --calib calib/reaper.txt \
    --buffer 1024 \
    --filename-is-timestamp \
    --disable_vis