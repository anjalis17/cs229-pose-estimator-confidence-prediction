#!/usr/bin/env bash
#
# Run inference on all samples in each split.
#
# Usage:  bash scripts/run_test.sh            # all three splits
#         bash scripts/run_test.sh synthetic  # one split
#
set -eo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO/spnv2/tools"
CFG="../experiments/offline_train_full_config_phi3_BN.yaml"
DROOT="$REPO/data"
CAM="$REPO/data/camera.json"
KPS="$REPO/spnv2/core/utils/models/tangoPoints.mat"

csv_for() {
  case "$1" in
    synthetic) echo "synthetic/labels/validation.csv" ;;
    lightbox)  echo "lightbox/labels/test.csv" ;;
    sunlamp)   echo "sunlamp/labels/test.csv" ;;
    *) echo "unknown split: $1" >&2; exit 1 ;;
  esac
}

if [ "$#" -gt 0 ]; then SPLITS="$*"; else SPLITS="synthetic lightbox sunlamp"; fi

for s in $SPLITS; do
  echo "================= $s ================="
  # MAX_SAMPLES=[TODO: n] python3 test.py --cfg "$CFG" \
python3 test.py --cfg "$CFG" \
    DATASET.ROOT "$DROOT" \
    DATASET.CAMERA "$CAM" \
    DATASET.KEYPOINTS "$KPS" \
    TEST.TEST_CSV "$(csv_for "$s")"
done

echo
echo ">>> Done. Predictions at:"
echo "    spnv2/tools/outputs/efficientdet_d3/full_config/<domain>/predictions_pose.mat"
echo ">>> Assess:  python scripts/assess_features.py"
