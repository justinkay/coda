import argparse
import json
import os
import subprocess


def main():
    p = argparse.ArgumentParser(description="Launch epsilon grid search for missing tasks")
    p.add_argument("--pred-dir", default="data", help="Directory containing prediction tensors")
    p.add_argument("--results", default="best_epsilons.json", help="Path to best epsilon results json")
    args = p.parse_args()

    existing = set()
    if os.path.exists(args.results):
        with open(args.results) as f:
            data = json.load(f)
        for k in data.keys():
            base = k[:-3] if k.endswith(".pt") else k
            existing.add(base)

    pt_files = [f for f in os.listdir(args.pred_dir) if f.endswith(".pt") and not f.endswith("_labels.pt")]
    pt_files.sort()
    srun_prefix = [
        "srun",
        "-p",
        "vision-beery",
        "-q",
        "vision-beery-main",
        "-t",
        "7-00:00:00",
        "--mem=64GB",
        "--cpus-per-task",
        "16",
        "--gpus-per-node",
        "1",
    ]

    for fname in pt_files:
        task = fname[:-3]
        if task in existing:
            continue
        cmd = srun_prefix + [
            "python",
            "modelselector_eps_gridsearch_v2.py",
            "--task",
            task,
            "--pred-dir",
            args.pred_dir,
        ]
        print("Launching:", " ".join(cmd))
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
