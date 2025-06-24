import argparse
import os
import subprocess
import mlflow

# warning: database lock issues can occur if using db for logging
USE_DB = False
if USE_DB:
    mlflow.set_tracking_uri('sqlite:///coda.sqlite')


def seed_run_status(task, method, seed):
    run_name = f"{task}-{method}-{seed}"
    runs = mlflow.search_runs(
        experiment_names=[task],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1,
    )
    if len(runs) == 0:
        return None, False
    finished = runs.status.values[0] == 'FINISHED'
    stochastic = (
        'params.stochastic' in runs.columns and
        runs['params.stochastic'].values[0] == 'True'
    )
    return finished, stochastic


def run_needed(task, method, max_seeds):
    status = seed_run_status(task, method, 0)
    if status[0] is None:
        return True  # seed0 never run
    finished0, stochastic0 = status
    if not finished0:
        return True
    if not stochastic0:
        return False
    for seed in range(1, max_seeds):
        finished, _ = seed_run_status(task, method, seed)
        if not finished:
            return True
    return False


def main():
    p = argparse.ArgumentParser(description='Launch all methods for all tasks')
    p.add_argument('--pred-dir', default='data', help='Directory containing prediction tensors')
    p.add_argument('--methods', default='iid,activetesting,vma,model_picker,uncertainty,coda',
                   help='Comma-separated list of methods to run')
    p.add_argument('--seeds', type=int, default=5, help='Maximum number of seeds')
    args = p.parse_args()

    tasks = [f[:-3] for f in os.listdir(args.pred_dir)
             if f.endswith('.pt') and not f.endswith('_labels.pt')]
    tasks.sort()
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]

    srun_prefix = [
        'srun', '-p', 'vision-beery', '-q', 'vision-beery-main',
        '-t', '7-00:00:00', '--mem=64GB', '--cpus-per-task', '16',
        '--gpus-per-node', '1'
    ]

    procs = []
    for task in tasks:
        for method in methods:
            if not run_needed(task, method, args.seeds):
                print(f"Skipping {task}/{method}; all seeds finished")
                continue
            cmd = srun_prefix + [
                'python', 'main.py',
                '--task', task,
                '--method', method,
                '--data-dir', args.pred_dir,
                '--seeds', str(args.seeds)
            ]
            print('Launching:', ' '.join(cmd))
            procs.append(subprocess.Popen(cmd))

    for p in procs:
        p.wait()


if __name__ == '__main__':
    main()
