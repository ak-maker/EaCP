"""Run all ImageNet-C experiments: 4 corruptions x 5 severities."""
import subprocess
import sys

UPDATES = "none eacp eacp_gini eacp_top2 eacp_gini_norm eacp_top2_norm eacp_adaptive eacp_online eacp_sliding eacp_top2_norm_adaptive eacp_adaptive_scaling tent_ecp tent_ecp_adaptive"

corruptions = ["contrast", "brightness", "gaussian_noise", "motion_blur"]

for corr in corruptions:
    for sev in range(1, 6):
        print(f"\n{'='*50}")
        print(f"Running {corr} severity {sev}")
        print(f"{'='*50}")
        cmd = [
            sys.executable, "main.py",
            "--dataset", "imagenet-c",
            "--corruption", corr,
            "--severity", str(sev),
            "--model", "resnet50",
            "--save-name", f"table2_{corr}_s{sev}",
            "--alpha", "0.1",
            "--updates", *UPDATES.split()
        ]
        subprocess.run(cmd)

print("\nAll ImageNet-C experiments done!")
