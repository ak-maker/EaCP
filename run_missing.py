"""Run all missing experiments sequentially."""
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

UPDATES = "none eacp eacp_gini eacp_top2 eacp_gini_norm eacp_top2_norm eacp_adaptive eacp_online eacp_sliding eacp_top2_norm_adaptive eacp_adaptive_scaling tent_ecp tent_ecp_adaptive".split()

def run(args):
    print(f"\n>>> {' '.join(args)}")
    subprocess.run([sys.executable, "main.py"] + args)

# ImageNet-R (if missing)
r_path = "results/imagenet-r/table1_r.csv"
if not os.path.exists(r_path):
    print("=" * 50 + "\nRunning ImageNet-R\n" + "=" * 50)
    run(["--dataset", "imagenet-r", "--model", "resnet50", "--save-name", "table1_r",
         "--alpha", "0.1", "--updates"] + UPDATES)

# ImageNet-C missing experiments
for corr in ["contrast", "brightness", "gaussian_noise", "motion_blur"]:
    for sev in range(1, 6):
        csv_path = f"results/imagenet-c/{corr}/table2_{corr}_s{sev}.csv"
        if not os.path.exists(csv_path):
            print(f"{'='*50}\nRunning {corr} severity {sev}\n{'='*50}")
            run(["--dataset", "imagenet-c", "--corruption", corr, "--severity", str(sev),
                 "--model", "resnet50", "--save-name", f"table2_{corr}_s{sev}",
                 "--alpha", "0.1", "--updates"] + UPDATES)
        else:
            print(f"Skipping {corr} s{sev} (already exists)")

print("\nAll missing experiments done!")
