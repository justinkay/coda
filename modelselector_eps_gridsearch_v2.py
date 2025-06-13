import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm

from datasets import Dataset
from coda.baselines.modelpicker import ModelPicker


def majority_vote_labels(preds: torch.Tensor) -> np.ndarray:
    """Return majority vote labels for predictions of shape (H,N,C)."""
    pred_classes = preds.argmax(dim=2)  # (H,N)
    votes = pred_classes.transpose(0, 1).cpu().numpy()  # (N,H)
    maj = np.zeros(votes.shape[0], dtype=int)
    for i in range(votes.shape[0]):
        vals, cnts = np.unique(votes[i], return_counts=True)
        maj[i] = vals[np.argmax(cnts)]
    return maj


def create_realisations(num_items: int, num_reals: int, pool_size: int) -> np.ndarray:
    """Random subsets of indices for each realisation."""
    return np.array([np.random.permutation(num_items)[:pool_size] for _ in range(num_reals)])


def calculate_model_ranking(predictions: np.ndarray, oracle: np.ndarray):
    """Return model ranking and accuracies for given predictions and oracle."""
    if oracle.ndim == 1:
        oracle = oracle[:, None]
    accuracies = np.sum(predictions == oracle, axis=0)
    ranking = np.argsort(-accuracies, kind="stable")
    return ranking, accuracies / len(oracle)


def run_realisation(preds_hnc: torch.Tensor, oracle: np.ndarray, epsilon: float, budget: int, seed: int):
    """Run ModelPicker on a single realisation."""
    device = preds_hnc.device
    H, N, C = preds_hnc.shape
    subset_ds = type("Tmp", (), {})()
    subset_ds.preds = preds_hnc
    subset_ds.device = device
    selector = ModelPicker(subset_ds, epsilon=epsilon)
    ranking_t = []
    for _ in range(budget):
        idx, p = selector.get_next_item_to_label()
        label = int(oracle[idx])
        selector.add_label(idx, label, p)
        ranking_t.append(selector.get_best_model_prediction())
    pred_classes = preds_hnc.argmax(dim=2).transpose(0, 1).cpu().numpy()
    gt_ranking, gt_acc = calculate_model_ranking(pred_classes, oracle)
    return ranking_t, (gt_ranking, gt_acc)


def evaluate(success_runs, acc_runs):
    success_mean = np.mean(success_runs, axis=0)
    acc_mean = np.mean(acc_runs, axis=0)
    return success_mean, acc_mean


def run_grid_search(pred_path: str, args):
    """Run the epsilon grid search for a single prediction tensor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Dataset(pred_path, device)
    _, N, _ = data.preds.shape
    majority = majority_vote_labels(data.preds)

    pool_size = min(args.pool_size, N)
    budget = min(args.budget, pool_size)
    realisations = create_realisations(N, args.iterations, pool_size)

    eps_list = [float(e) for e in args.epsilons.split(",")]
    results = {}
    for eps in eps_list:
        success_runs = []
        acc_runs = []
        for i, r in enumerate(tqdm(realisations, desc=f"eps={eps}")):
            subset_preds = data.preds[:, r, :]
            subset_oracle = majority[r]
            ranking_t, gt = run_realisation(
                subset_preds,
                subset_oracle,
                eps,
                budget,
                seed=i,
            )
            best_acc = np.max(gt[1])
            best_models = np.where(gt[1] == best_acc)[0]
            success_t = [int(idx in best_models) for idx in ranking_t]
            acc_t = [gt[1][idx] for idx in ranking_t]
            success_runs.append(success_t)
            acc_runs.append(acc_t)
        success_mean, acc_mean = evaluate(np.array(success_runs), np.array(acc_runs))
        avg_success = float(np.mean(success_mean))
        try:
            t_fast = int(np.argmax(success_mean >= args.threshold))
            if success_mean[t_fast] < args.threshold:
                t_fast = float("inf")
        except ValueError:
            t_fast = float("inf")
        results[eps] = {
            "success_mean": success_mean.tolist(),
            "acc_mean": acc_mean.tolist(),
            "avg_success": avg_success,
            "fastest_t": t_fast,
        }
        print(f"eps={eps:.3f} avg_success={avg_success:.3f} fastest_t={t_fast}")

    best_avg = max(results.items(), key=lambda x: x[1]["avg_success"])[0]
    best_fast = min(results.items(), key=lambda x: x[1]["fastest_t"])[0]
    print("\nOptimal epsilon (avg_success):", best_avg)
    print("Optimal epsilon (fastest):", best_fast)
    return {
        "best_avg": best_avg,
        "best_fast": best_fast,
        "metrics": results,
    }


def main():
    p = argparse.ArgumentParser(description="Unsupervised epsilon tuning via grid search (reference implementation)")
    p.add_argument("--preds", help="Path to (H,N,C) tensor of model predictions")
    p.add_argument("--pred-dir", default='data', help="Directory containing prediction tensors (.pt)")
    p.add_argument("--task", help="Task name; uses <task>.pt from --pred-dir", default=None)
    p.add_argument("--epsilons", default="0.35,0.36,0.37,0.38,0.39,0.40,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49")
    # Match the reference implementation defaults (Table 3)
    p.add_argument("--iterations", type=int, default=1000,
                   help="Number of random realisations")
    p.add_argument("--pool-size", type=int, default=1000,
                   help="Realisation pool size")
    p.add_argument("--budget", type=int, default=1000,
                   help="Budget per realisation")
    p.add_argument("--threshold", type=float, default=0.9, help="Success threshold for fastest metric")
    args = p.parse_args()

    if args.task:
        args.preds = os.path.join(args.pred_dir, args.task + ".pt")

    if not args.preds and not args.pred_dir:
        p.error("Either --preds, --pred-dir or --task must be specified")

    results_path = "best_epsilons.json"
    overall = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            overall = json.load(f)

    if args.task or args.preds:
        if args.task:
            key = args.task
            path = args.preds
        else:
            key = os.path.basename(args.preds)
            path = args.preds
        if key in overall:
            print(key, "already computed; skipping")
        else:
            res = run_grid_search(path, args)
            overall[key] = {"best_avg": res["best_avg"], "best_fast": res["best_fast"]}
            with open(results_path, "w") as f:
                json.dump(overall, f, indent=2)
    elif args.pred_dir:
        pt_files = [f for f in os.listdir(args.pred_dir) if f.endswith(".pt") and not f.endswith("_labels.pt")]
        pt_files.sort()
        for fname in pt_files:
            if fname in overall:
                print(fname, "already computed; skipping")
                continue
            path = os.path.join(args.pred_dir, fname)
            res = run_grid_search(path, args)
            overall[fname] = {"best_avg": res["best_avg"], "best_fast": res["best_fast"]}
            with open(results_path, "w") as f:
                json.dump(overall, f, indent=2)
    else:
        p.error("No predictions specified")


if __name__ == "__main__":
    main()