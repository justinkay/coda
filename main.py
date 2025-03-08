LOGGING = 'local' # 'wandb'

VERSION = 0.1 # 0.0

if LOGGING == 'comet':
    import comet_ml # import this first to avoid warnings
elif LOGGING == 'wandb':
    import wandb
else:
    import csv

import argparse
import random
import numpy as np
import torch
from tqdm import tqdm

# from experiment import initialize_experiment_wandb, initialize_experiment_comet
from options import DATASETS, LOSS_FNS, ACCURACY_FNS, ORACLES
from ams.iid import IID
from ams.bb import BB
from ams.dirichlet import Dirichlet
from ams.baselines import ActiveTesting, VMA, ModelPicker, Uncertainty


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
    parser.add_argument("--dataset", help="{ 'domainnet126', ... } ", default=None)
    parser.add_argument("--task", help="{ 'sketch_painting', ... }", default=None)
    parser.add_argument("--best-epochs-only", action='store_true', help="Keep only best checkpoint from each hparam run")
    # parser.add_argument("--best-hparams-only", action='store_true', help="Keep only best hyperparameter setting from each algorithm")

    # benchmarking settings
    parser.add_argument("--acc", help="Accuracy fn. Options specific to dataset, see main.py.", default="acc")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--subsample-pct", type=int, help="Percentage of runs to analyze",  default=100)
    parser.add_argument("--force-reload", action='store_true', help="Load directly from feature files rather than large dat file.")
    parser.add_argument("--no-write", action='store_true', help="Don't write preds to an intermediate .dat or .pt file.")
    parser.add_argument("--no-comet", action='store_true', help="Disable logging with Comet ML")
    parser.add_argument("--no-wandb", action='store_true', help="Disable logging with wandb")
    parser.add_argument("--filter-bad", action='store_true', help="Filter bad models using the oracle")
    parser.add_argument("--seeds", type=int, default=5) # how many seeds to use - one experiment per seed
    parser.add_argument("--name", help="Readable name for the experiment. If blank, will concatenate args.", default="")
    parser.add_argument("--force-rerun", action="store_true", help="Overwrite existing comet runs.")
    if VERSION > 0.0:
        parser.add_argument("--log-every", action="store_true")
        parser.add_argument("--log-dir", default="my_logs_0305/")

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
    true_losses = oracle.true_losses(dataset.pred_logits)

    # experiment = None
    if LOGGING == 'wandb':
        experiment = initialize_experiment_wandb(args, dataset, seed)
    elif LOGGING == 'comet':
        experiment = initialize_experiment_comet(args, dataset, seed)
    # if experiment is None: return

    best_loss = min(oracle.true_losses(dataset.pred_logits))
    # experiment.log({"Best loss": best_loss})
    # experiment.log({"All (true) losses": true_losses.cpu()})
    # print("Best possible loss", best_loss)
    
    # metrics we will track
    cumulative_regret_loss = 0
    
    # initialize method
    if args.method == 'iid':
        selector = IID(dataset, loss_fn, 
                        prefilter_fn=args.prefilter_fn,
                        prefilter_n=args.prefilter_n)
    elif args.method == 'uncertainty':
        selector = Uncertainty(dataset, loss_fn)
    elif args.method == 'beta':
        selector = BB(dataset, 
                      prior_source=args.prior_source, 
                      prior_strength=args.prior_strength,
                      q=args.q,
                      select=args.select,
                      stochastic=args.stochastic,
                      importance_weighting=args.importance_weighting,
                      prefilter_fn=args.prefilter_fn,
                      prefilter_n=args.prefilter_n,
                      epsilon=args.epsilon,
                      item_prior_source=args.item_priors,
                      update_strength=args.update_strength)
    elif args.method == 'dirichlet':
        selector = Dirichlet(dataset,
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
        from ams.baselines.modelpicker import TASK_EPS
        if args.task not in TASK_EPS.keys():
            print(args.task, "not in TASK_EPS; using default")
        epsilon = args.epsilon if args.epsilon > 0.0 else TASK_EPS[args.task]
        selector = ModelPicker(dataset, 
                               epsilon=epsilon,
                               prior_source=args.prior_source,
                               item_prior_source=args.item_priors)

    # active model selection loop
    regrets = []
    cum_regrets = []
    for m in tqdm(range(args.iters)):
    
        # select item, label, select model
        chosen_idx, selection_prob = selector.get_next_item_to_label()
        true_class = labeler(chosen_idx)
        selector.add_label(chosen_idx, true_class, selection_prob)
        best_model_idx_pred = selector.get_best_model_prediction()

        ### Log metrics ###
        # experiment.log({"Pred. best model idx": best_model_idx_pred}, step=m+1)
        # experiment.log({"Pred. best model, true loss": true_losses[best_model_idx_pred]}, step=m+1)

        regret_loss = true_losses[best_model_idx_pred] - best_loss
        cumulative_regret_loss += regret_loss
        print("Regret at", m, ":", regret_loss)

        regrets.append(regret_loss.item())
        cum_regrets.append(cumulative_regret_loss.item())

        if VERSION > 0.0 and args.log_every:
            if LOGGING == 'wandb':
                experiment.log({"Regret (loss)": regret_loss}, step=m+1)
                experiment.log({"Cumulative regret (loss)": cumulative_regret_loss}, step=m+1)
            elif LOGGING == 'comet':
                experiment.log_metrics({"Regret (loss)": regret_loss}, step=m+1)
                experiment.log_metrics({"Cumulative regret (loss)": cumulative_regret_loss}, step=m+1)

        # Loss estimation error 
        # risk_estimates = selector.get_risk_estimates()
        # if risk_estimates is not None:
        #     loss_errors = torch.abs(true_losses - risk_estimates)
        #     mean_loss_error = loss_errors.mean()
            # experiment.log({"Mean absolute loss estimation error": mean_loss_error}, step=m+1)
            # experiment.log_metrics({"Mean absolute loss estimation error": mean_loss_error}, step=m+1)

        # to_log = selector.get_additional_to_log()
        # if to_log is not None:
        #     if LOGGING == 'wandb':
        #         experiment.log(to_log, step=m+1)
        #     elif LOGGING == 'comet':
        #         experiment.log_metrics(to_log, step=m+1)

    if LOGGING == 'wandb':
        experiment.log({f"cumulative_regret_{args.iters}": cumulative_regret_loss,
                        "all_regrets": regrets,
                        "all_cum_regrets": cum_regrets,
                        })
    elif LOGGING == 'comet':
        experiment.log_metrics({f"cumulative_regret_{args.iters}": cumulative_regret_loss,
                        "all_regrets": regrets,
                        "all_cum_regrets": cum_regrets,})
    elif LOGGING == 'local':
        alg_detail_exclude = [
                                "best_epochs_only",
                                "acc",
                                "iters",
                                "subsample_pct",
                                "force_reload",
                                "no_write",
                                "no_wandb",
                                "filter_bad",
                                "seeds",
                                "name",
                                "force_rerun",
                                "importance_weighting",
                                "select",
                                "update_rule",
                                "log_every",
                                "loss",
                                # 'q',
                                'epsilon',
                                'update_strength',
                                'base_strength',
                                'dataset_filter',
                                'no_comet',
                                'prior_strength',
                                'stochastic',
                                'prior_source',
                                'item_priors',
                                'prefilter_fn',
                                'log_dir',
                            ]

        uncertainty_exclude = [
            'dataset_filter',
            'prefilter_fn',
            'prefilter_n',
            'epsilon',
            'update_strength',
            'base_strength',
            'base_prior',
            'temperature',
            'alpha',
            'learning_rate_ratio',
        ]
        
        all_exclude = alg_detail_exclude +  uncertainty_exclude if args.method=="uncertainty" else alg_detail_exclude

        file_name =  "-".join([f"{k}={v}" for k, v in vars(args).items() if k not in all_exclude]) + f"-seed={seed}" # for consistency
        file_name = file_name.replace("/", "|")
        header = ['regret', 'cumulative regret']
        data = zip(regrets, cum_regrets)
        with open(f'{args.log_dir}/{file_name}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for row in data:
                writer.writerow(row)

    if LOGGING == 'wandb':
        wandb.finish()

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    # Load prediction results of all hypotheses
    dataset = DATASETS[args.dataset](args.task, use_target_val=True) #, dataset_filter=args.dataset_filter)
    dataset.load_runs(subsample_pct=args.subsample_pct, force_reload=args.force_reload, write=not args.no_write)
    H, N, C = dataset.pred_logits.shape
    print("dataset.pred_logits.shape before filtering", dataset.pred_logits.shape)

    # Loss and accuracy functions
    accuracy_fn = ACCURACY_FNS[args.dataset][args.acc].to(dataset.device)
    loss_fn = LOSS_FNS[args.loss]

    # Create oracle
    oracle = ORACLES[args.dataset](dataset, task=args.task, loss_fn=loss_fn, accuracy_fn=accuracy_fn)

    # Filter dataset if desired
    dataset.filter(oracle, args.best_epochs_only, 
                    best_hparams_only=args.dataset == 'domainnet126', # TODO come back to this for big run
                    filter_bad=args.filter_bad)
    
    # Model selection loop
    for seed in range(args.seeds):
        print("Running active model selection with seed", seed)
        print("DEBUG ARGS", args.__dict__)
        do_model_selection_experiment(dataset, oracle, args, loss_fn, seed=seed)

if __name__ == "__main__":
    main()