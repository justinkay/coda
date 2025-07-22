import argparse
import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient

# Use same configuration as main.py
USE_DB = True
if USE_DB:
    mlflow.set_tracking_uri('sqlite:///coda.sqlite')

client = MlflowClient()


def confirm(prompt):
    resp = input(prompt + ' [y/N] ')
    return resp.lower() in {'y', 'yes'}


def parse_list(option):
    if not option:
        return None
    return [x.strip() for x in option.split(',') if x.strip()]


def delete_all(skip_confirm=False):
    targets = []
    if os.path.exists('coda.sqlite'):
        targets.append('coda.sqlite')
    if os.path.exists('mlruns'):
        targets.append('mlruns')
    if not targets:
        print('Database already empty.')
        return
    if not skip_confirm and not confirm(f"Are you sure you want to delete {', '.join(targets)}?"):
        print('Aborted.')
        return
    for path in targets:
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
    print('Deleted', ', '.join(targets))


def delete_selected(tasks, methods, skip_confirm=False):
    print("deleting tasks:", tasks)
    print("deleting methods", methods)
    experiments = client.search_experiments()
    to_delete_runs = []
    to_delete_exps = []
    
    print(f"Found {len(experiments)} experiments to check...")
    
    for i, exp in enumerate(experiments):
        print(f"Processing experiment {i+1}/{len(experiments)}: {exp.name}")
        if tasks is not None and exp.name not in tasks:
            continue
        if methods is None:
            to_delete_exps.append(exp)
        else:
            # Get all runs in this experiment
            all_runs = client.search_runs([exp.experiment_id])
            print(f"  Found {len(all_runs)} runs in experiment {exp.name}")
            for run in all_runs:
                # Check if any of the methods appear in the run name
                # TODO: check for exact match
                if any(method in run.info.run_name for method in methods):
                    print(f"    Marking run for deletion: {run.info.run_name}")
                    to_delete_runs.append(run)

    if not to_delete_runs and not to_delete_exps:
        print('Nothing found to delete.')
        return
    summary = []
    for exp in to_delete_exps:
        summary.append(f"experiment {exp.name}")
    for run in to_delete_runs:
        summary.append(f"run {run.info.run_name}")
    if not skip_confirm and not confirm('Are you sure you want to delete ' + ', '.join(summary) + '?'):
        print('Aborted.')
        return
    for exp in to_delete_exps:
        client.delete_experiment(exp.experiment_id)
    for run in to_delete_runs:
        client.delete_run(run.info.run_id)
    print('Deletion complete.')


def main():
    p = argparse.ArgumentParser(description='Clear mlflow database entries.')
    p.add_argument('--tasks', help='Comma-separated list of tasks to clear')
    p.add_argument('--methods', help='Comma-separated list of methods to clear')
    p.add_argument('--yes', action='store_true', help='Skip confirmation prompt')
    args = p.parse_args()

    tasks = parse_list(args.tasks)
    methods = parse_list(args.methods)

    if tasks is None and methods is None:
        delete_all(args.yes)
    else:
        delete_selected(tasks, methods, args.yes)


if __name__ == '__main__':
    main()
