import argparse
import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('sqlite:///coda.sqlite')
client = MlflowClient()


def confirm(prompt):
    resp = input(prompt + ' [y/N] ')
    return resp.lower() in {'y', 'yes'}


def parse_list(option):
    if not option:
        return None
    return [x.strip() for x in option.split(',') if x.strip()]


def delete_all():
    targets = []
    if os.path.exists('coda.sqlite'):
        targets.append('coda.sqlite')
    if os.path.exists('mlruns'):
        targets.append('mlruns')
    if not targets:
        print('Database already empty.')
        return
    if not confirm(f"Are you sure you want to delete {', '.join(targets)}?"):
        print('Aborted.')
        return
    for path in targets:
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
    print('Deleted', ', '.join(targets))


def delete_selected(tasks, methods):
    experiments = client.search_experiments()
    to_delete_runs = []
    to_delete_exps = []
    for exp in experiments:
        if tasks is not None and exp.name not in tasks:
            continue
        if methods is None:
            to_delete_exps.append(exp)
        else:
            for method in methods:
                runs = client.search_runs([exp.experiment_id],
                                          filter_string=f"params.method = '{method}'")
                to_delete_runs.extend(runs)
    if not to_delete_runs and not to_delete_exps:
        print('Nothing found to delete.')
        return
    summary = []
    for exp in to_delete_exps:
        summary.append(f"experiment {exp.name}")
    for run in to_delete_runs:
        summary.append(f"run {run.info.run_id}")
    if not confirm('Are you sure you want to delete ' + ', '.join(summary) + '?'):
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
    args = p.parse_args()

    tasks = parse_list(args.tasks)
    methods = parse_list(args.methods)

    if tasks is None and methods is None:
        delete_all()
    else:
        delete_selected(tasks, methods)


if __name__ == '__main__':
    main()
