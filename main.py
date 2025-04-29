# LOGGING = 'local' # 'wandb'

VERSION = 0.2 # 0.0

# if LOGGING == 'comet':
#     import comet_ml # import this first to avoid warnings
# elif LOGGING == 'wandb':
#     import wandb
# else:
#     import csv
import mlflow

import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
import os
from torchmetrics import Accuracy

# from experiment import initialize_experiment_wandb, initialize_experiment_comet
from options import LOSS_FNS, ACCURACY_FNS
from datasets import Dataset
from oracle import Oracle

from coda import CODA
from coda.baselines import IID, ActiveTesting, VMA, ModelPicker, Uncertainty



def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    # dataset settings
    # parser.add_argument("--dataset", help="{ 'domainnet126', ... } ", default=None)
    parser.add_argument("--task", help="{ 'sketch_painting', ... }", default=None)
    # parser.add_argument("--best-epochs-only", action='store_true', help="Keep only best checkpoint from each hparam run")
    # parser.add_argument("--best-hparams-only", action='store_true', help="Keep only best hyperparameter setting from each algorithm")
    parser.add_argument("--data-dir", default='data')

    # benchmarking settings
    parser.add_argument("--acc", help="Accuracy fn. Options specific to dataset, see main.py.", default="acc")
    parser.add_argument("--iters", type=int, default=100)
    # parser.add_argument("--subsample-pct", type=int, help="Percentage of runs to analyze",  default=100)
    # parser.add_argument("--force-reload", action='store_true', help="Load directly from feature files rather than large dat file.")
    # parser.add_argument("--no-write", action='store_true', help="Don't write preds to an intermediate .dat or .pt file.")
    # parser.add_argument("--no-comet", action='store_true', help="Disable logging with Comet ML")
    # parser.add_argument("--no-wandb", action='store_true', help="Disable logging with wandb")
    # parser.add_argument("--filter-bad", action='store_true', help="Filter bad models using the oracle")
    parser.add_argument("--seeds", type=int, default=5) # how many seeds to use - one experiment per seed
    # parser.add_argument("--name", help="Readable name for the experiment. If blank, will concatenate args.", default="")
    parser.add_argument("--force-rerun", action="store_true", help="Overwrite existing comet runs.")
    if VERSION > 0.0:
        parser.add_argument("--log-every", action="store_true")
        parser.add_argument("--log-dir", default="my_logs_0305/")
    parser.add_argument("--experiment-name", default=None) # overrides default of using task as experiment name

    # method settings
    parser.add_argument("--loss", help="{ 'ce', 'acc', ... }", default="acc",)
    parser.add_argument("--method", help="{ 'iid', 'beta', 'activetesting', 'vma' }", default='iid')
    parser.add_argument("--prior-strength", type=float, default=10.0, help="Strength of beta distribution initial alpha/beta")
    parser.add_argument("--q", default="eig", help="{ 'iid', 'eig', 'l1', 'weighted_l1', 'max_regret', 'reduce_regret_local', 'reduce_regret_global' }")
    parser.add_argument("--select", default="sample", help="{ 'sample', 'means' }")
    parser.add_argument("--stochastic", action='store_true', help="Randomly select from q if True; greedy if False")
    parser.add_argument("--importance-weighting", action='store_true', help="Use importance weighting for posterior updates (only applicable if stochastic=True)")
    parser.add_argument("--prefilter-fn", default='disagreement', help="{ None, 'disagreement', 'iid' }")
    parser.add_argument("--prefilter-n", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--prior-source", default="ens-exp", help="{ 'ens-exp', 'ens-01', 'ds', 'ens-soft-01' }")
    parser.add_argument("--item-priors", default='ens', help="{ 'none', 'ens', bma-adaptive' }")
    parser.add_argument("--update-rule", default="hard", help="Hard or soft dirichlet confusion matrix updates")

    # deprecating
    if VERSION == 0.0:
        parser.add_argument("--update-strength", default=1.0, type=float, help="Multiplicative factor for beta posterior updates")
        parser.add_argument("--base-strength", default=1.0, type=float)
        parser.add_argument("--hypothetical-update-strength", default=1.0, type=float)
    else:
        parser.add_argument("--base-prior", default="diag")

    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--alpha", default=0.9, type=float)
    parser.add_argument("--learning-rate-ratio", default=0.01, type=float)

    return parser.parse_args()

def do_model_selection_experiment(dataset, oracle, args, loss_fn, seed=0):
    seed_all(seed)
    labeler = oracle # Label function - oracle or (future work) a user
    true_losses = oracle.true_losses(dataset.preds)
    best_loss = min(oracle.true_losses(dataset.preds))
    print("Best possible loss is", best_loss)

    # initialize method
    if args.method == 'iid':
        selector = IID(dataset, loss_fn)
    elif args.method == 'uncertainty':
        selector = Uncertainty(dataset, loss_fn)
    elif args.method == 'coda':
        selector = CODA(dataset,
                            prior_source=args.prior_source, 
                            q=args.q,
                            prefilter_fn=args.prefilter_fn,
                            prefilter_n=args.prefilter_n,
                            epsilon=args.epsilon,
                            update_rule=args.update_rule,
                            # update_strength=args.update_strength,
                            # prior_strength=args.prior_strength,
                            # base_strength=args.base_strength,
                            # hypothetical_update_strength=args.hypothetical_update_strength,
                            temperature=args.temperature,
                            alpha=args.alpha,
                            learning_rate_ratio=args.learning_rate_ratio,
                            base_prior=args.base_prior
                        )
    elif args.method == 'activetesting':
        selector = ActiveTesting(dataset, loss_fn)
    elif args.method == 'vma':
        selector = VMA(dataset, loss_fn)
    elif args.method == 'model_picker':
        from coda.baselines.modelpicker import TASK_EPS
        if args.task not in TASK_EPS.keys():
            print(args.task, "not in TASK_EPS; using default")
        epsilon = args.epsilon if args.epsilon > 0.0 else TASK_EPS[args.task]
        selector = ModelPicker(dataset, 
                               epsilon=epsilon,
                               prior_source=args.prior_source,
                               item_prior_source=args.item_priors)

    # active model selection loop
    cumulative_regret_loss = 0
    for m in tqdm(range(args.iters)):
        # select item, label, select model
        chosen_idx, selection_prob = selector.get_next_item_to_label()
        true_class = labeler(chosen_idx)
        selector.add_label(chosen_idx, true_class, selection_prob)
        best_model_idx_pred = selector.get_best_model_prediction()

        regret_loss = true_losses[best_model_idx_pred] - best_loss
        cumulative_regret_loss += regret_loss
        print("Regret at", m, ":", regret_loss)

        mlflow.log_metric("regret", regret_loss.item(), step=m+1)
        mlflow.log_metric("cumulative regret", cumulative_regret_loss.item(), step=m+1)

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    # Load prediction results of all hypotheses
    dataset = Dataset(os.path.join(args.data_dir, args.task + ".pt"), device=device)

    # Create oracle
    accuracy_fn = Accuracy(task="multiclass", num_classes=126, average="micro") # TODO
    loss_fn = LOSS_FNS[args.loss]
    oracle = Oracle(dataset, loss_fn=loss_fn, accuracy_fn=accuracy_fn)
    
    ## Model selection loop
    # create mlflow 'experiment' (= dataset/task)
    experiment_name = args.experiment_name or args.task
    mlflow.set_experiment(experiment_name)
    
    def get_mlflow_run_id(run_name):
        run_id = None
        matching_runs = mlflow.search_runs(experiment_names=['sketch_painting'], filter_string=f"tags.mlflow.runName = '{run_name}'", max_results=1)
        if len(matching_runs): 
            run_id = matching_runs.run_id.values[0]
            # TODO: status = matching_runs.status.values[0]; check for 'FINISHED'
        return run_id

    # create mlflow 'run' (= algorithm)
    # check for existing and overwrite
    # TODO: Could give option to not overwrite
    run_name = "-".join([experiment_name, args.method])
    run_id = get_mlflow_run_id(run_name)
    with mlflow.start_run(run_id=run_id, run_name=run_name):                                              
        mlflow.log_params(args.__dict__)
        for seed in range(args.seeds):
            # create nested ml flow 'run' (= seed)
            seed_run_name = "-".join([experiment_name, args.method, str(seed)])
            seed_run_id = get_mlflow_run_id(seed_run_name)
            with mlflow.start_run(nested=True, run_id=seed_run_id, run_name=seed_run_name):                                         
                mlflow.log_param("seed", seed)
                print("Running active model selection with seed", seed)
                print("DEBUG ARGS", args.__dict__)
                do_model_selection_experiment(dataset, oracle, args, loss_fn, seed=seed)

if __name__ == "__main__":
    main()