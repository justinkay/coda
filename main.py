import mlflow
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
import os
from torchmetrics import Accuracy

from coda import CODA
from coda.baselines import IID, ActiveTesting, VMA, ModelPicker, Uncertainty
from coda.datasets import Dataset
from coda.options import LOSS_FNS
from coda.oracle import Oracle
from logging_util import plot_bar

DEBUG_VIZ = True

mlflow.set_tracking_uri('sqlite:///coda.sqlite')

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
    parser.add_argument("--task", help="{ 'sketch_painting', ... }", default=None)
    parser.add_argument("--data-dir", default='data')

    # benchmarking settings
    parser.add_argument("--acc", help="Accuracy fn. Options specific to dataset, see main.py.", default="acc")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=5) # how many seeds to use - one experiment per seed
    parser.add_argument("--force-rerun", action="store_true", help="Overwrite existing runs.")
    parser.add_argument("--experiment-name", default=None) # overrides default of using task as experiment name

    # method settings
    parser.add_argument("--loss", help="{ 'ce', 'acc', ... }", default="acc",)
    parser.add_argument("--method", help="{ 'iid', 'beta', 'activetesting', 'vma' }", default='iid')
    parser.add_argument("--q", default="eig", help="{ 'iid', 'eig', 'l1', 'weighted_l1', 'max_regret', 'reduce_regret_local', 'reduce_regret_global' }")
    parser.add_argument("--prefilter-fn", default='disagreement', help="{ None, 'disagreement', 'iid' }")
    parser.add_argument("--prefilter-n", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--prior-source", default="ens-exp", help="{ 'ens-exp', 'ens-01', 'ds', 'ens-soft-01' }")
    parser.add_argument("--item-priors", default='ens', help="{ 'none', 'ens', bma-adaptive' }")
    
    # CODA settings
    parser.add_argument("--base-prior", default="diag")
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--alpha", default=0.9, type=float)
    parser.add_argument("--learning-rate-ratio", default=0.01, type=float)

    return parser.parse_args()

def do_model_selection_experiment(dataset, oracle, args, loss_fn, seed=0):
    seed_all(seed)
    true_losses = oracle.true_losses(dataset.preds)
    best_loss = min(oracle.true_losses(dataset.preds))
    print("Best possible loss is", best_loss)

    # initialize method
    if args.method == 'iid':
        selector = IID(dataset, loss_fn)
    elif args.method == 'uncertainty':
        selector = Uncertainty(dataset, loss_fn)
    elif args.method.startswith('coda'):
        selector = CODA.from_args(dataset, args)
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
    else:
        raise ValueError(args.method + " is not a supported method.")

    # Get prior regret
    best_model_idx_pred = selector.get_best_model_prediction(step=0)# TODO CODA HACK STEP
    regret_loss = true_losses[best_model_idx_pred] - best_loss
    print("Regret at 0:", regret_loss)

    ## Active model selection loop
    cumulative_regret_loss = 0
    for m in tqdm(range(args.iters)):
        # select item, label, select model
        chosen_idx, selection_prob = selector.get_next_item_to_label(step=m+1) # TODO CODA HACK STEP
        true_class = oracle(chosen_idx)
        selector.add_label(chosen_idx, true_class, selection_prob)
        best_model_idx_pred = selector.get_best_model_prediction(step=m+1) # TODO CODA HACK STEP

        # compute and log metrics
        regret_loss = true_losses[best_model_idx_pred] - best_loss
        cumulative_regret_loss += regret_loss
        print("Regret at", m+1, ":", regret_loss)
        mlflow.log_metric("regret", regret_loss.item(), step=m+1)
        mlflow.log_metric("cumulative regret", cumulative_regret_loss.item(), step=m+1)

    return selector.stochastic

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
        matching_runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string=f"tags.mlflow.runName = '{run_name}'", max_results=1)
        finished = False
        stochastic = None
        if len(matching_runs): 
            run_id = matching_runs.run_id.values[0]
            finished = matching_runs.status.values[0] == 'FINISHED'
            stochastic = 'params.stochastic' in matching_runs.columns and matching_runs['params.stochastic'].values[0] == 'True'
        return run_id, finished, stochastic

    # create mlflow 'run' (= algorithm)
    run_name = "-".join([experiment_name, args.method])
    run_id, _, _ = get_mlflow_run_id(run_name)
    with mlflow.start_run(run_id=run_id, run_name=run_name):                                              
        mlflow.log_params(args.__dict__)
        for seed in range(args.seeds):
            # create nested ml flow 'run' (= seed)
            seed_run_name = "-".join([experiment_name, args.method, str(seed)])
            seed_run_id, seed_finished, seed_stochastic = get_mlflow_run_id(seed_run_name)
            if seed_finished:
                print("Seed", seed, "finished. Skipping.")
            else:
                with mlflow.start_run(nested=True, run_id=seed_run_id, run_name=seed_run_name):                                         
                    mlflow.log_param("seed", seed)
                    print("Running active model selection with seed", seed)
                    print("DEBUG ARGS", args.__dict__)
                    seed_stochastic = do_model_selection_experiment(dataset, oracle, args, loss_fn, seed=seed)
                    mlflow.log_param("stochastic", seed_stochastic)
            
            if not seed_stochastic:
                print("Method is not stochastic for this task. Skipping further seeds.")
                break

if __name__ == "__main__":
    main()