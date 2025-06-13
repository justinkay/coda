"""Grid search for the :class:`ModelPicker` epsilon parameter.

This script mirrors the unsupervised epsilon tuning procedure from the
`model-selector` project.  For a given set of candidate epsilon values we run
``ModelPicker`` with a *noisy oracle* obtained by taking the majority vote of
all available models.  After each run the final selected model is compared to
that majority vote on the **entire** dataset and the epsilon giving the highest
agreement is reported.

Example
-------

```
python scripts/model_picker_epsilon_gridsearch.py \
    --preds path/to/predictions.pt --epsilons 0.35,0.40,0.45 --iters 100
```
"""

import argparse
import random
import numpy as np
import torch
from tqdm import tqdm

from coda.datasets import Dataset
from coda.baselines import ModelPicker
from main import seed_all

# If not in the same directory, adjust imports as needed

def majority_vote_oracle(dataset):
    """
    Creates a 1D array of shape (N,) that is the "majority vote" across the H models
    for each item. This is used as our self-supervised "oracle."
    
    dataset.preds => shape (H,N,C)
    We'll do a discrete prediction for each model => shape(H,N).
    Then majority vote across H for each item => shape(N,).
    """
    device = dataset.preds.device
    H, N, C = dataset.preds.shape

    # Discrete predictions => shape(H,N)
    pred_classes_hn = dataset.preds.argmax(dim=2)
    
    # transpose => shape(N,H)
    predictions_nh = pred_classes_hn.transpose(0,1).cpu().numpy()

    majority_labels = np.zeros(N, dtype=int)
    for i in range(N):
        # count the most common label among the H predictions
        votes, counts = np.unique(predictions_nh[i], return_counts=True)
        maj = votes[np.argmax(counts)]
        majority_labels[i] = maj

    return majority_labels

def self_supervised_oracle_factory(majority_labels):
    """
    Returns a function "oracle(idx)" that yields majority_labels[idx].
    This simulates a labeler that is just the majority vote.
    """
    def oracle_fn(idx):
        return majority_labels[idx]
    return oracle_fn

def measure_final_agreement(best_model_idx, dataset, majority_labels):
    """
    Once we pick a final best model, measure how often that model's predictions
    match the majority vote across ALL items in the dataset. 
    This is the final "success" in self-supervised approach.
    """
    device = dataset.preds.device
    H, N, C = dataset.preds.shape

    # gather discrete predictions from best_model_idx => shape(N,)
    preds = dataset.preds[best_model_idx]
    pred_classes = preds.argmax(dim=1).cpu().numpy()

    # compare to majority_labels => shape(N,)
    matches = (pred_classes == majority_labels)
    agreement = np.mean(matches)
    return agreement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True, help="Path to a tensor of shape (H,N,C) containing model predictions")
    parser.add_argument("--epsilons", type=str, default="0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49",
                        help="Comma-separated list of epsilons to try.")
    parser.add_argument("--iters", type=int, default=100,
                        help="Number of steps in active model selection.")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--name", default="", help="Experiment name")
    parser.add_argument("--best-hparams-only", action="store_true")
    args = parser.parse_args()

    # don't log these runs
    args.no_wandb = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is:", device)

    # parse epsilons
    eps_values = [float(x.strip()) for x in args.epsilons.split(",")]

    # 1) Load dataset
    dataset = Dataset(args.preds, device)
    H, N, C = dataset.preds.shape
    print("Loaded predictions with shape", dataset.preds.shape)

    # 2) Build a "majority vote" self-supervised oracle
    majority_labels = majority_vote_oracle(dataset)

    def self_supervised_oracle(idx):
        return majority_labels[idx]

    # Loop over epsilons
    results = {}
    for eps in eps_values:
        print(f"\n=== Epsilon = {eps} ===")
        seed_agreements = []
        for s in range(args.seeds):
            print("  seed:", s)
            seed_all(s)

            best_model_idx = run_experiment_and_capture_best_model(
                dataset, self_supervised_oracle, eps, args.iters, seed=s
            )

            agreement = measure_final_agreement(best_model_idx, dataset, majority_labels)
            print(f"  [seed={s}] final model {best_model_idx}, agreement={agreement:.3f}")
            seed_agreements.append(agreement)

        mean_agree = np.mean(seed_agreements)
        print(f"Eps={eps}, avg agreement over seeds => {mean_agree:.3f}")
        results[eps] = mean_agree

    # pick best epsilon
    best_epsilon = max(results, key=results.get)
    best_score = results[best_epsilon]
    print("\n======= Grid Search Summary =======")
    for e, v in results.items():
        print(f"Eps={e:.3f}, Score={v:.3f}")
    print(f"BEST => eps={best_epsilon:.3f}, Score={best_score:.3f}")

    del dataset
    torch.cuda.empty_cache()


def run_experiment_and_capture_best_model(dataset, oracle_fn, epsilon, iters, seed=0):
    """Run ``ModelPicker`` for ``iters`` steps and return the selected model."""

    from main import seed_all
    seed_all(seed)

    # build a ModelPicker for this epsilon
    selector = ModelPicker(dataset, epsilon=epsilon)

    # We'll do the same loop of 'args.iters'
    # track the best model each iteration, but we only need the final one
    best_model_idx = None
    # for m in range(args.iters): 
    for m in tqdm(range(iters), desc=f"Eps={epsilon}, seed={seed}"):
        chosen_idx, selection_prob = selector.get_next_item_to_label()
        # get label from our self-supervised oracle
        true_class = oracle_fn(chosen_idx)
        selector.add_label(chosen_idx, true_class, selection_prob)
        best_model_idx = selector.get_best_model_prediction()

        # We skip all the logging to W&B or comet to keep it minimal 
        # (or you can copy that logic if you prefer)

    return best_model_idx


if __name__ == "__main__":
    main()
