#!/bin/bash
# Run all experiments for the course project
# Usage: bash run_all_experiments.sh

UPDATES="none eacp eacp_gini eacp_top2 eacp_gini_norm eacp_top2_norm eacp_adaptive eacp_online eacp_sliding eacp_top2_norm_adaptive eacp_adaptive_scaling tent_ecp tent_ecp_adaptive"

echo "============================================"
echo "Table 1: Stationary shifts"
echo "============================================"

# ImageNet-V2 (already have results, but rerun for consistency)
echo ">>> Running ImageNet-V2..."
python main.py --dataset imagenet-v2 --model resnet50 --save-name table1_v2 --alpha 0.1 \
  --updates $UPDATES

# ImageNet-R
echo ">>> Running ImageNet-R..."
python main.py --dataset imagenet-r --model resnet50 --save-name table1_r --alpha 0.1 \
  --updates $UPDATES

# ImageNet-A
echo ">>> Running ImageNet-A..."
python main.py --dataset imagenet-a --model resnet50 --save-name table1_a --alpha 0.1 \
  --updates $UPDATES

echo "============================================"
echo "Table 2: ImageNet-C corruptions"
echo "============================================"

# 4 corruption types x 5 severity levels
for CORR in contrast brightness gaussian_noise motion_blur; do
  for SEV in 1 2 3 4 5; do
    echo ">>> Running ImageNet-C: $CORR severity $SEV..."
    python main.py --dataset imagenet-c --corruption $CORR --severity $SEV \
      --model resnet50 --save-name "table2_${CORR}_s${SEV}" --alpha 0.1 \
      --updates $UPDATES
  done
done

echo "============================================"
echo "All experiments complete!"
echo "Results saved in results/ directory"
echo "============================================"
