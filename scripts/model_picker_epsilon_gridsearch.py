"""
model_picker_epsilon_gridsearch.py

A self-contained script that performs a self-supervised grid search over epsilon
for your ModelPicker class, using the "majority vote" across all models as a
noisy oracle. It then picks the epsilon that yields the highest final agreement
with that majority on all items.

Usage:
  python model_picker_epsilon_gridsearch.py --dataset <DATASET> --epsilons 0.1,0.2,0.3 --seeds 3 ...

Notes:
  - This script re-implements your "ModelPicker" approach with different epsilons,
    using a self-supervised "oracle" that is just the item-wise majority vote of
    all model predictions. We then run the active model selection loop (like in
    main.py) for each epsilon and measure final success as "which model is picked
    as best" and "how often does that model match majority on all items." 
  - The best epsilon is the one that yields the highest such agreement.
"""

import argparse
import random
import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

# Reuse your code from main.py or other modules:
from options import DATASETS, LOSS_FNS, ACCURACY_FNS, ORACLES
from coda.baselines import ModelPicker
from main import seed_all, do_model_selection_experiment

# If not in the same directory, adjust imports as needed

def majority_vote_oracle(dataset):
    """
    Creates a 1D array of shape (N,) that is the "majority vote" across the H models
    for each item. This is used as our self-supervised "oracle."
    
    dataset.pred_logits => shape (H,N,C)
    We'll do a discrete prediction for each model => shape(H,N).
    Then majority vote across H for each item => shape(N,).
    """
    device = dataset.pred_logits.device
    H, N, C = dataset.pred_logits.shape

    # Convert logits to discrete predictions => shape(H,N)
    pred_probs = F.softmax(dataset.pred_logits, dim=2)  # => (H,N,C)
    pred_classes_hn = pred_probs.argmax(dim=2)         # => (H,N)
    
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
    device = dataset.pred_logits.device
    H, N, C = dataset.pred_logits.shape

    # gather discrete predictions from best_model_idx => shape(N,)
    pred_probs = F.softmax(dataset.pred_logits[best_model_idx].unsqueeze(0), dim=2)
    # shape(1,N,C)
    pred_classes = pred_probs.argmax(dim=2).squeeze(0).cpu().numpy()  # shape(N,)

    # compare to majority_labels => shape(N,)
    matches = (pred_classes == majority_labels)
    agreement = np.mean(matches)
    return agreement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", nargs='+', help="{ 'sketch_painting', ... }", default=None)
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

    for task in args.task:

        # 1) Load dataset
        ds_class = DATASETS[args.dataset]
        dataset = ds_class(task, use_target_val=True)
        dataset.load_runs()
        # just for filtering
        accuracy_fn = ACCURACY_FNS[args.dataset]['acc'].to(dataset.device)
        loss_fn = LOSS_FNS['acc']
        oracle = ORACLES[args.dataset](dataset, task=task, loss_fn=loss_fn, accuracy_fn=accuracy_fn)
        dataset.filter(oracle, best_epochs_only=False, 
                       best_hparams_only=args.dataset=='domainnet126', 
                       filter_bad=False)
        H, N, C = dataset.pred_logits.shape
        print("Loaded dataset:", args.dataset, "with shape", dataset.pred_logits.shape)

        # 2) Build a "majority vote" self-supervised oracle
        majority_labels = majority_vote_oracle(dataset)
        # We'll create a function: oracle(idx) => majority_labels[idx]
        def self_supervised_oracle(idx):
            return majority_labels[idx]

        # 3) We'll define a local "loss_fn" that is 'acc' or 'ce' or so, 
        #    but it's not super relevant because we won't be measuring real regret 
        #    with a ground-truth. We'll pass it anyway for consistency:
        # If your do_model_selection_experiment needs an actual oracle object with .true_losses(...),
        # that might cause trouble. We'll keep it minimal:
        def dummy_loss_fn(pred_label, true_label):
            # not used
            return 0.0

        # We loop over each epsilon, run do_model_selection_experiment for each seed,
        # measure final "agreement" => how well the final chosen model matches the majority over all items,
        # average across seeds => store => pick best epsilon.
        results = {}
        for eps in eps_values:
            print(f"\n=== Epsilon = {eps} ===")
            # We'll store final agreements for each seed
            seed_agreements = []
            for s in range(args.seeds):
                print("  seed:", s)
                seed_all(s)

                # We'll define local args for do_model_selection_experiment
                class LocalArgs:
                    pass
                local_args = LocalArgs()
                local_args.dataset = args.dataset
                local_args.task = task
                local_args.method = 'model_picker'
                local_args.iters = args.iters
                local_args.seeds = 1
                local_args.no_wandb = args.no_wandb
                local_args.force_reload = False
                local_args.best_epochs_only = False
                local_args.best_hparams_only = False
                local_args.filter_bad = False
                local_args.acc = "acc"
                local_args.loss = "acc"
                local_args.q = None
                local_args.select = None
                local_args.stochastic = False
                local_args.importance_weighting = False
                local_args.prefilter_fn = None
                local_args.prefilter_n = 500
                local_args.prior_strength = 10.0
                local_args.name = args.name
                local_args.epsilon = eps
                local_args.force_rerun = False

                # We'll run do_model_selection_experiment with our self_supervised_oracle
                # But we must pass that oracle instead of the real dataset-based oracle.
                # We'll also pass a dummy loss_fn since we don't rely on the real one.
                # do_model_selection_experiment returns None, but it logs to W&B, etc. 
                # We just want to get the final best model from the selector. 
                # => We'll modify do_model_selection_experiment to return the best model idx 
                #    or we do a small hack: we define a function wrapper that captures it.

                # Hack: We'll wrap the original do_model_selection_experiment 
                # so we can retrieve the final best model:
                best_model_idx = run_experiment_and_capture_best_model(
                    dataset, self_supervised_oracle, local_args, dummy_loss_fn, seed=s
                )

                # measure final agreement
                agreement = measure_final_agreement(best_model_idx, dataset, majority_labels)
                print(f"  [seed={s}] final model {best_model_idx}, agreement={agreement:.3f}")
                seed_agreements.append(agreement)

            # store average
            mean_agree = np.mean(seed_agreements)
            print(f"Eps={eps}, avg agreement over seeds => {mean_agree:.3f}")
            results[eps] = mean_agree

        # pick best epsilon
        best_epsilon = max(results, key=results.get)
        best_score = results[best_epsilon]
        print("\n======= Grid Search Summary =======")
        for e,v in results.items():
            print(f"Eps={e:.3f}, Score={v:.3f}")
        print(f"BEST => eps={best_epsilon:.3f}, Score={best_score:.3f}")

        del dataset
        torch.cuda.empty_cache()


def run_experiment_and_capture_best_model(dataset, oracle_fn, args, loss_fn, seed=0):
    """
    Wraps do_model_selection_experiment but returns the final best model index from the selector.
    We'll do that by monkey-patching the final lines of do_model_selection_experiment,
    or we replicate do_model_selection_experiment logic here in a shortened form.
    """
    # We'll replicate the core of do_model_selection_experiment from main.py, 
    # but skipping the real oracle references:
    from main import seed_all
    seed_all(seed)

    # build a ModelPicker for this epsilon
    # but we do the same approach as do_model_selection_experiment
    from ams.baselines import ModelPicker
    # We'll define a minimal approach:
    # note: dataset is your "logged" dataset with pred_logits, etc.
    selector = ModelPicker(dataset, epsilon=args.epsilon)

    # We'll do the same loop of 'args.iters'
    # track the best model each iteration, but we only need the final one
    best_model_idx = None
    # for m in range(args.iters): 
    for m in tqdm(range(args.iters), desc=f"Eps={args.epsilon}, seed={seed}"):
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
