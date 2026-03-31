#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Required arguments
# ---------------------------------------------------------------------------
INPUT_DIR=""
CALIB=""
OUTPUT_DIR=""

# ---------------------------------------------------------------------------
# Tunable parameters – defaults tuned for a calibrated monocular camera
# ---------------------------------------------------------------------------
WEIGHTS="${SCRIPT_DIR}/droid.pth"
STRIDE=2            # process every frame
BUFFER=1024         # max keyframes in sliding window
CAMERA_MODEL="radtan"

# Motion filter: minimum optical-flow magnitude before a frame is considered
FILTER_THRESH=1.75

# Keyframe selection: lower = keep more keyframes (better accuracy, more memory)
KEYFRAME_THRESH=2.0

WARMUP=8

# Frontend graph
FRONTEND_THRESH=16.0
FRONTEND_WINDOW=25
FRONTEND_RADIUS=2
FRONTEND_NMS=1

# Backend loop-closure / global BA
BACKEND_THRESH=22.0
BACKEND_RADIUS=2
BACKEND_NMS=3

BETA=0.3

DISABLE_VIS=true    # set to true to disable OpenGL visualizer (requires PyOpenGL and GLFW)
UPSAMPLE=true       # high-res disparity needed for point cloud export

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 --input-dir DIR --calib FILE --output-dir DIR [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --input-dir     DIR    directory of input images (filenames = ns timestamps)"
    echo "  --calib         FILE   calibration file (fx fy cx cy [dist...])"
    echo "  --output-dir    DIR    folder for reconstruction.pt and reconstruction.ply"
    echo ""
    echo "Optional:"
    echo "  --weights       FILE   model weights           (default: droid.pth)"
    echo "  --stride        INT    frame stride            (default: ${STRIDE})"
    echo "  --buffer        INT    keyframe buffer size    (default: ${BUFFER})"
    echo "  --camera-model  STR    radtan | fisheye        (default: ${CAMERA_MODEL})"
    echo "  --filter-thresh FLOAT  motion filter threshold (default: ${FILTER_THRESH})"
    echo "  --keyframe-thresh FLOAT keyframe pruning threshold (default: ${KEYFRAME_THRESH})"
    echo "  --disable-vis          disable OpenGL visualizer"
    echo "  --no-upsample          skip high-res disparity upsampling"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir)      INPUT_DIR="$2";          shift 2 ;;
        --calib)          CALIB="$2";              shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";         shift 2 ;;
        --weights)        WEIGHTS="$2";            shift 2 ;;
        --stride)         STRIDE="$2";             shift 2 ;;
        --buffer)         BUFFER="$2";             shift 2 ;;
        --camera-model)   CAMERA_MODEL="$2";       shift 2 ;;
        --filter-thresh)  FILTER_THRESH="$2";      shift 2 ;;
        --keyframe-thresh) KEYFRAME_THRESH="$2";   shift 2 ;;
        --disable-vis)    DISABLE_VIS=true;        shift 1 ;;
        --no-upsample)    UPSAMPLE=false;          shift 1 ;;
        -h|--help)        usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ -z "$INPUT_DIR" || -z "$CALIB" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: --input-dir, --calib, and --output-dir are required."
    echo ""
    usage
fi

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
CMD=(
    python3 "${SCRIPT_DIR}/droid-slam.py"
    --input-dir      "$INPUT_DIR"
    --calib          "$CALIB"
    --output-dir     "$OUTPUT_DIR"
    --weights        "$WEIGHTS"
    --stride         "$STRIDE"
    --buffer         "$BUFFER"
    --camera-model   "$CAMERA_MODEL"
    --beta           "$BETA"
    --filter_thresh  "$FILTER_THRESH"
    --warmup         "$WARMUP"
    --keyframe_thresh "$KEYFRAME_THRESH"
    --frontend_thresh  "$FRONTEND_THRESH"
    --frontend_window  "$FRONTEND_WINDOW"
    --frontend_radius  "$FRONTEND_RADIUS"
    --frontend_nms     "$FRONTEND_NMS"
    --backend_thresh   "$BACKEND_THRESH"
    --backend_radius   "$BACKEND_RADIUS"
    --backend_nms      "$BACKEND_NMS"
)

[[ "$DISABLE_VIS" == true ]] && CMD+=(--disable_vis)
[[ "$UPSAMPLE"    == true ]] && CMD+=(--upsample)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "Running DROID-SLAM:"
echo ""

exec "${CMD[@]}"
