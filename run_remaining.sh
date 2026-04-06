#!/bin/bash
# Run all remaining experiments sequentially.
# Usage: tmux new -s exp && conda activate EaCP && bash run_remaining.sh
cd /home/jliu222/EaCP

UPDATES="none eacp eacp_gini eacp_top2 eacp_gini_norm eacp_top2_norm eacp_adaptive eacp_online eacp_sliding eacp_top2_norm_adaptive eacp_adaptive_scaling tent_ecp tent_ecp_adaptive"

# Kill any lingering experiment processes
pkill -f "run_imagenet_c.py" 2>/dev/null
sleep 2

echo "============================================"
echo "Starting remaining experiments at $(date)"
echo "============================================"

# 1. ImageNet-V2 (rerun for clean results)
if [ ! -f results/imagenet-v2/table1_v2.csv ]; then
  echo ">>> [$(date +%H:%M)] ImageNet-V2..."
  python main.py --dataset imagenet-v2 --model resnet50 --save-name table1_v2 --alpha 0.1 --updates $UPDATES
fi

# 2. ImageNet-R
if [ ! -f results/imagenet-r/table1_r.csv ]; then
  echo ">>> [$(date +%H:%M)] ImageNet-R..."
  python main.py --dataset imagenet-r --model resnet50 --save-name table1_r --alpha 0.1 --updates $UPDATES
fi

# 3. ImageNet-C: contrast, brightness, motion_blur (gaussian_noise already done)
for CORR in contrast brightness motion_blur; do
  for SEV in 1 2 3 4 5; do
    CSV="results/imagenet-c/${CORR}/table2_${CORR}_s${SEV}.csv"
    if [ ! -f "$CSV" ]; then
      echo ">>> [$(date +%H:%M)] ImageNet-C: $CORR severity $SEV..."
      python main.py --dataset imagenet-c --corruption $CORR --severity $SEV \
        --model resnet50 --save-name "table2_${CORR}_s${SEV}" --alpha 0.1 --updates $UPDATES
    else
      echo ">>> Skipping $CORR s$SEV (exists)"
    fi
  done
done

echo "============================================"
echo "All done at $(date)"
echo "============================================"
