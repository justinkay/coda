import argparse
import os
import subprocess
import mlflow
import time
import re

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

def get_running_jobs():
    """Get list of running SLURM job IDs for current user."""
    try:
        result = subprocess.run(['squeue', '-u', os.getenv('USER', ''), '-h', '-o', '%i'], 
                              capture_output=True, text=True, check=True)
        job_ids = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        return job_ids
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

def launch_job_with_tracking(cmd, job_queue, running_jobs):
    """Launch a job and track it."""
    print('Launching:', ' '.join(cmd))
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Give the job a moment to start and get a SLURM job ID
        time.sleep(2)
        
        # Try to get job ID from squeue (this is approximate)
        current_jobs = get_running_jobs()
        new_jobs = set(current_jobs) - set(running_jobs.keys())
        
        if new_jobs:
            job_id = list(new_jobs)[0]  # Take the first new job
            running_jobs[job_id] = {
                'cmd': cmd,
                'proc': proc,
                'task': cmd[cmd.index('--task') + 1] if '--task' in cmd else 'unknown',
                'method': cmd[cmd.index('--method') + 1] if '--method' in cmd else 'unknown'
            }
            return job_id
        else:
            # Fallback to process tracking if we can't get SLURM job ID
            fake_id = f"proc_{proc.pid}"
            running_jobs[fake_id] = {
                'cmd': cmd,
                'proc': proc,
                'task': cmd[cmd.index('--task') + 1] if '--task' in cmd else 'unknown',
                'method': cmd[cmd.index('--method') + 1] if '--method' in cmd else 'unknown'
            }
            return fake_id
    except Exception as e:
        print(f"Error launching job: {e}")
        return None

def check_job_completion(running_jobs):
    """Check which jobs have completed and remove them from tracking."""
    current_jobs = get_running_jobs()
    completed = []
    
    for job_id, job_info in list(running_jobs.items()):
        if job_id.startswith('proc_'):
            # Process-based tracking
            if job_info['proc'].poll() is not None:
                completed.append(job_id)
        else:
            # SLURM-based tracking
            if job_id not in current_jobs:
                completed.append(job_id)
    
    for job_id in completed:
        job_info = running_jobs.pop(job_id)
        print(f"Job {job_id} completed: {job_info['task']}/{job_info['method']}")
    
    return len(completed)


def main():
    p = argparse.ArgumentParser(description='Launch all methods for all tasks')
    p.add_argument('--pred-dir', default='data', help='Directory containing prediction tensors')
    p.add_argument('--methods', default='iid,activetesting,vma,model_picker,uncertainty,coda',
                   help='Comma-separated list of methods to run')
    p.add_argument('--seeds', type=int, default=5, help='Maximum number of seeds')
    p.add_argument('--max-concurrent-jobs', type=int, default=16, 
                   help='Maximum number of concurrent SLURM jobs (default: 16)')
    p.add_argument('--polling-interval', type=int, default=30,
                   help='Seconds between job status checks (default: 30)')
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
    
    # Build job queue
    job_queue = []
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
            job_queue.append(cmd)
    
    if not job_queue:
        print("No jobs to run!")
        return
    
    print(f"Found {len(job_queue)} jobs to run with max {args.max_concurrent_jobs} concurrent jobs")
    
    # Job management with throttling
    running_jobs = {}  # job_id -> job_info
    job_index = 0
    
    # Launch initial batch
    while job_index < len(job_queue) and len(running_jobs) < args.max_concurrent_jobs:
        cmd = job_queue[job_index]
        job_id = launch_job_with_tracking(cmd, job_queue, running_jobs)
        if job_id:
            job_index += 1
    
    # Monitor and launch remaining jobs
    while job_index < len(job_queue) or running_jobs:
        time.sleep(args.polling_interval)
        
        # Check for completed jobs
        completed_count = check_job_completion(running_jobs)
        
        # Launch new jobs if slots available
        while job_index < len(job_queue) and len(running_jobs) < args.max_concurrent_jobs:
            cmd = job_queue[job_index]
            job_id = launch_job_with_tracking(cmd, job_queue, running_jobs)
            if job_id:
                job_index += 1
        
        # Progress report
        total_jobs = len(job_queue)
        completed_jobs = job_index - len(running_jobs)
        running_count = len(running_jobs)
        pending_count = total_jobs - job_index
        
        print(f"Progress: {completed_jobs}/{total_jobs} completed, "
              f"{running_count} running, {pending_count} pending")
        
        if running_jobs:
            print(f"Running jobs: {list(running_jobs.keys())}")
    
    print("All jobs completed!")


if __name__ == '__main__':
    main()
