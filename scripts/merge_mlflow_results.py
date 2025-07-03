#!/usr/bin/env python3
"""
Merge MLflow results from individual job directories into a single tracking store.
Run this after all SLURM jobs complete.
"""

import os
import shutil
import glob
import mlflow
from mlflow.tracking import MlflowClient
import argparse
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_child_metrics_batch(client: MlflowClient, child_run_ids: List[str], metric_keys: List[str]) -> Dict[str, Dict[str, List]]:
    """Batch fetch metrics for multiple child runs."""
    def fetch_run_metrics(run_id):
        run_metrics = {}
        for metric in metric_keys:
            try:
                history = client.get_metric_history(run_id, metric)
                run_metrics[metric] = [(point.step, point.value) for point in history]
            except mlflow.exceptions.MlflowException:
                run_metrics[metric] = []
        return run_id, run_metrics
    
    all_metrics = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_run = {executor.submit(fetch_run_metrics, run_id): run_id for run_id in child_run_ids}
        for future in as_completed(future_to_run):
            run_id, metrics = future.result()
            all_metrics[run_id] = metrics
    
    return all_metrics

def aggregate_metrics(client: MlflowClient, metric_keys: Optional[List[str]] = None) -> None:
    """Compute step-wise means for metric_keys and log them to parent runs."""
    
    if metric_keys is None:
        metric_keys = ["regret", "cumulative regret"]
    
    # Get all experiments and runs in a single batch
    experiments = client.search_experiments()
    
    for exp in experiments:
        exp_id = exp.experiment_id
        
        # Get all runs at once
        all_runs = mlflow.search_runs(
            experiment_ids=[exp_id], output_format="pandas", max_results=5000
        )
        if all_runs.empty:
            continue
            
        # Separate parent and child runs
        if "tags.mlflow.parentRunId" in all_runs.columns:
            parent_mask = all_runs["tags.mlflow.parentRunId"].isna()
            parent_runs = all_runs[parent_mask]
            child_runs = all_runs[~parent_mask]
        else:
            parent_runs = all_runs
            child_runs = all_runs.iloc[0:0]  # Empty dataframe
        
        if child_runs.empty:
            continue
            
        # Group child runs by parent
        parent_to_children = defaultdict(list)
        for _, child_row in child_runs.iterrows():
            parent_id = child_row["tags.mlflow.parentRunId"]
            parent_to_children[parent_id].append(child_row["run_id"])
        
        # Process each parent and its children
        for _, parent_row in parent_runs.iterrows():
            parent_run_id = parent_row["run_id"]
            child_run_ids = parent_to_children.get(parent_run_id, [])
            
            if not child_run_ids:
                continue
                
            # Batch fetch all metrics for all children
            all_child_metrics = get_child_metrics_batch(client, child_run_ids, metric_keys)
            
            # Aggregate metrics by step
            collector = defaultdict(lambda: defaultdict(list))
            for run_id, run_metrics in all_child_metrics.items():
                for metric, step_values in run_metrics.items():
                    for step, value in step_values:
                        collector[metric][step].append(value)
            
            # Log aggregated metrics in batches
            batch_metrics = []
            for metric, step_dict in collector.items():
                for step, values in step_dict.items():
                    if not values:
                        continue
                    mean_val = float(np.mean(values))
                    std_val = float(np.std(values)) if len(values) > 1 else 0.0
                    batch_metrics.extend([
                        (f"mean_{metric}", mean_val, step),
                        (f"std_{metric}", std_val, step)
                    ])
                    print(
                        f"[Exp {exp.name}] parent {parent_run_id[:8]} | "
                        f"step {step} mean_{metric} = {mean_val:.6f} ± {std_val:.6f} "
                        f"(n={len(values)})"
                    )
            
            # Batch log metrics
            for key, value, step in batch_metrics:
                client.log_metric(parent_run_id, key=key, value=value, step=step)

def copy_run_metrics_batch(source_client: MlflowClient, run_id: str, metric_keys: List[str]) -> Dict[str, List]:
    """Batch copy all metrics for a single run."""
    all_metrics = {}
    for key in metric_keys:
        try:
            metric_history = source_client.get_metric_history(run_id, key)
            all_metrics[key] = [(m.value, m.step, m.timestamp) for m in metric_history]
        except:
            all_metrics[key] = []
    return all_metrics

def merge_mlflow_runs():
    """Merge all mlruns_job_* directories into a single mlruns directory."""
    
    # Set up destination tracking URI
    dest_dir = os.path.abspath('./mlruns_merged')
    dest_tracking_uri = f'file://{dest_dir}'
    mlflow.set_tracking_uri(dest_tracking_uri)
    dest_client = MlflowClient(dest_tracking_uri)
    
    # Find all job directories
    job_dirs = glob.glob('mlruns_job_*')
    print(f"Found {len(job_dirs)} job directories to merge")
    
    experiment_mapping = {}
    
    for job_dir in sorted(job_dirs):
        print(f"Processing {job_dir}...")
        
        source_tracking_uri = f'file://{os.path.abspath(job_dir)}'
        source_client = MlflowClient(source_tracking_uri)
        
        try:
            experiments = source_client.search_experiments()
            
            for exp in experiments:
                if exp.name == "Default":
                    continue
                    
                # Create or get experiment in destination
                try:
                    dest_exp = dest_client.get_experiment_by_name(exp.name)
                    dest_exp_id = dest_exp.experiment_id
                except:
                    dest_exp_id = dest_client.create_experiment(exp.name)
                
                experiment_mapping[exp.experiment_id] = dest_exp_id
                
                # Get all runs from this experiment at once
                runs = source_client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=5000
                )
                
                # Separate parent and child runs
                parent_runs = [r for r in runs if 'mlflow.parentRunId' not in r.data.tags]
                child_runs = [r for r in runs if 'mlflow.parentRunId' in r.data.tags]
                
                # Get all unique metric keys upfront
                all_metric_keys = set()
                for run in runs:
                    all_metric_keys.update(run.data.metrics.keys())
                all_metric_keys = list(all_metric_keys)
                
                run_mapping = {}
                
                # Copy parent runs with batch metric fetching
                for run in parent_runs:
                    print(f"  Copying run: {run.info.run_name}")
                    
                    with mlflow.start_run(
                        experiment_id=dest_exp_id,
                        run_name=run.info.run_name,
                        nested=False
                    ) as new_run:
                        run_mapping[run.info.run_id] = new_run.info.run_id
                        
                        # Batch copy parameters
                        for key, value in run.data.params.items():
                            mlflow.log_param(key, value)
                        
                        # Batch copy metrics
                        metrics_data = copy_run_metrics_batch(source_client, run.info.run_id, all_metric_keys)
                        for key, metric_points in metrics_data.items():
                            for value, step, timestamp in metric_points:
                                mlflow.log_metric(key, value, step=step, timestamp=timestamp)
                        
                        # Copy non-system tags
                        for key, value in run.data.tags.items():
                            if not key.startswith('mlflow.'):
                                mlflow.set_tag(key, value)
                
                # Copy child runs
                for run in child_runs:
                    print(f"  Copying child run: {run.info.run_name}")
                    parent_run_id = run.data.tags.get('mlflow.parentRunId')
                    
                    if parent_run_id in run_mapping:
                        dest_parent_id = run_mapping[parent_run_id]
                        
                        with mlflow.start_run(run_id=dest_parent_id):
                            with mlflow.start_run(
                                experiment_id=dest_exp_id,
                                run_name=run.info.run_name,
                                nested=True
                            ) as new_run:
                                run_mapping[run.info.run_id] = new_run.info.run_id
                                
                                # Batch copy parameters and metrics
                                for key, value in run.data.params.items():
                                    mlflow.log_param(key, value)
                                
                                metrics_data = copy_run_metrics_batch(source_client, run.info.run_id, all_metric_keys)
                                for key, metric_points in metrics_data.items():
                                    for value, step, timestamp in metric_points:
                                        mlflow.log_metric(key, value, step=step, timestamp=timestamp)
                                
                                for key, value in run.data.tags.items():
                                    if not key.startswith('mlflow.'):
                                        mlflow.set_tag(key, value)
                    else:
                        print(f"    Warning: Parent run {parent_run_id} not found for child {run.info.run_name}")
                                
        except Exception as e:
            print(f"Error processing {job_dir}: {e}")
            continue
    
    print(f"Merge complete! Results saved to: {dest_tracking_uri}")
    return dest_client

def cleanup_job_directories():
    """Remove individual job directories after successful merge."""
    job_dirs = glob.glob('mlruns_job_*')
    for job_dir in job_dirs:
        try:
            shutil.rmtree(job_dir)
            print(f"Removed {job_dir}")
        except Exception as e:
            print(f"Could not remove {job_dir}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Merge MLflow results from individual job directories and aggregate metrics'
    )
    parser.add_argument('--cleanup', action='store_true', 
                       help='Remove job directories after merge')
    parser.add_argument('--metrics', type=str, nargs='*',
                       default=['regret', 'cumulative regret'],
                       help='Metrics to aggregate (default: regret, cumulative regret)')
    args = parser.parse_args()
    
    dest_client = merge_mlflow_runs()
    
    if dest_client:
        print("Aggregating child run metrics into parent runs...")
        aggregate_metrics(dest_client, args.metrics)
        print("You can view with: mlflow ui --backend-store-uri ./mlruns_merged")
    
    if args.cleanup:
        cleanup_job_directories()