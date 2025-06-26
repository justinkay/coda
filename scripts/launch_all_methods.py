import argparse
import os
import subprocess
import mlflow

# Each job will use its own tracking directory
# Check merged results to determine what needs to be run

def check_completed_runs():
    """Check merged results to see what's already completed."""
    completed = set()
    
    # Check if merged results exist
    merged_dir = './mlruns_merged'
    if os.path.exists(merged_dir):
        try:
            mlflow.set_tracking_uri(f'file://{os.path.abspath(merged_dir)}')
            experiments = mlflow.search_experiments()
            
            for exp in experiments:
                if exp.name == "Default":
                    continue
                    
                runs = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string="status = 'FINISHED'",
                    max_results=1000
                )
                
                for _, run in runs.iterrows():
                    run_name = run.get('tags.mlflow.runName', '')
                    if run_name:
                        completed.add(run_name)
        except Exception as e:
            print(f"Could not read merged results: {e}")
    
    return completed

def run_needed(task, method, max_seeds, completed_runs):
    """Check if this task/method combination needs to be run."""
    
    # Check seed 0 first
    seed0_name = f"{task}-{method}-0"
    if seed0_name not in completed_runs:
        return True
    
    # If seed 0 exists, check if method is stochastic by looking for seed 1
    seed1_name = f"{task}-{method}-1" 
    if seed1_name not in completed_runs:
        # Might be non-stochastic (only needs seed 0) or needs more seeds
        # For safety, assume we need to check all seeds
        pass
    
    # Check all seeds up to max_seeds
    for seed in range(max_seeds):
        seed_name = f"{task}-{method}-{seed}"
        if seed_name not in completed_runs:
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

    # Check what's already completed
    completed_runs = check_completed_runs()
    print(f"Found {len(completed_runs)} completed runs in merged results")
    
    procs = []
    for task in tasks:
        for method in methods:
            if not run_needed(task, method, args.seeds, completed_runs):
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
