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

def aggregate_metrics(client: MlflowClient, metric_keys: Optional[List[str]] = None) -> None:
    """Compute step-wise means for metric_keys and log them to parent runs."""
    
    if metric_keys is None:
        metric_keys = ["regret", "cumulative regret"]
    
    # Iterate over every experiment
    for exp in client.search_experiments():
        exp_id = exp.experiment_id
        
        # Pull all runs in this experiment
        all_runs = mlflow.search_runs(
            experiment_ids=[exp_id], output_format="pandas"
        )
        if all_runs.empty:
            continue
            
        # Parent runs have no parentRunId tag (NaN in the dataframe)
        if "tags.mlflow.parentRunId" in all_runs.columns:
            parent_mask = all_runs["tags.mlflow.parentRunId"].isna()
            parent_runs = all_runs[parent_mask]
        else:
            # If no parentRunId column exists, all runs are parent runs
            parent_runs = all_runs
        
        for _, parent_row in parent_runs.iterrows():
            parent_run_id: str = parent_row["run_id"]
            
            # Child runs reference this parent via tag
            child_runs = mlflow.search_runs(
                experiment_ids=[exp_id],
                filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
                output_format="pandas",
            )
            if child_runs.empty:
                continue
                
            # Collector: {metric -> {step -> [values]}}
            collector: Dict[str, Dict[int, List[float]]] = {
                m: {} for m in metric_keys
            }
            
            for _, child_row in child_runs.iterrows():
                run_id = child_row["run_id"]
                for metric in metric_keys:
                    try:
                        history = client.get_metric_history(run_id, metric)
                    except mlflow.exceptions.MlflowException:
                        continue  # metric not found in this child
                    for point in history:
                        collector[metric].setdefault(point.step, []).append(point.value)
            
            # Log aggregated means back to the parent run
            for metric, step_dict in collector.items():
                for step, values in step_dict.items():
                    if not values:  # Skip empty value lists
                        continue
                    mean_val = float(np.mean(values))
                    std_val = float(np.std(values)) if len(values) > 1 else 0.0
                    client.log_metric(
                        parent_run_id,
                        key=f"mean_{metric}",
                        value=mean_val,
                        step=step,
                    )
                    client.log_metric(
                        parent_run_id,
                        key=f"std_{metric}",
                        value=std_val,
                        step=step,
                    )
                    print(
                        f"[Exp {exp.name}] parent {parent_run_id[:8]} | "
                        f"step {step} mean_{metric} = {mean_val:.6f} ± {std_val:.6f} "
                        f"(n={len(values)})"
                    )

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
    
    experiment_mapping = {}  # Map old experiment ID to new experiment ID
    
    for job_dir in sorted(job_dirs):
        print(f"Processing {job_dir}...")
        
        # Set up source client
        source_tracking_uri = f'file://{os.path.abspath(job_dir)}'
        source_client = MlflowClient(source_tracking_uri)
        
        try:
            # Get all experiments from this job
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
                
                # Get all runs from this experiment
                runs = source_client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=1000
                )
                
                # Separate parent and child runs
                parent_runs = []
                child_runs = []
                for run in runs:
                    if 'mlflow.parentRunId' in run.data.tags:
                        child_runs.append(run)
                    else:
                        parent_runs.append(run)
                
                # First pass: create parent runs
                run_mapping = {}  # old_run_id -> new_run_id
                
                for run in parent_runs:
                    print(f"  Copying run: {run.info.run_name}")
                    print(f"    Tags: {list(run.data.tags.keys())}")
                    if 'mlflow.parentRunId' in run.data.tags:
                        print(f"    Parent: {run.data.tags['mlflow.parentRunId']}")
                    
                    # Check if this is a nested run
                    parent_run_id = run.data.tags.get('mlflow.parentRunId')
                    
                    # Create new run in destination
                    with mlflow.start_run(
                        experiment_id=dest_exp_id,
                        run_name=run.info.run_name,
                        nested=parent_run_id is not None
                    ) as new_run:
                        run_mapping[run.info.run_id] = new_run.info.run_id
                        
                        # Copy parameters
                        for key, value in run.data.params.items():
                            mlflow.log_param(key, value)
                        
                        # Copy metrics
                        for key, metric_list in run.data.metrics.items():
                            # Get metric history
                            metric_history = source_client.get_metric_history(
                                run.info.run_id, key
                            )
                            for metric in metric_history:
                                mlflow.log_metric(
                                    key, 
                                    metric.value, 
                                    step=metric.step,
                                    timestamp=metric.timestamp
                                )
                        
                        # Copy tags (no parent relationship for parent runs)
                        for key, value in run.data.tags.items():
                            if not key.startswith('mlflow.'):  # Skip system tags
                                mlflow.set_tag(key, value)
                
                # Second pass: create child runs
                for run in child_runs:
                    print(f"  Copying child run: {run.info.run_name}")
                    parent_run_id = run.data.tags.get('mlflow.parentRunId')
                    
                    # Find the corresponding parent in destination
                    if parent_run_id in run_mapping:
                        dest_parent_id = run_mapping[parent_run_id]
                        
                        # Create nested run with explicit parent
                        with mlflow.start_run(
                            run_id=dest_parent_id
                        ):
                            with mlflow.start_run(
                                experiment_id=dest_exp_id,
                                run_name=run.info.run_name,
                                nested=True
                            ) as new_run:
                                run_mapping[run.info.run_id] = new_run.info.run_id
                                
                                # Copy parameters
                                for key, value in run.data.params.items():
                                    mlflow.log_param(key, value)
                                
                                # Copy metrics
                                for key, metric_list in run.data.metrics.items():
                                    metric_history = source_client.get_metric_history(
                                        run.info.run_id, key
                                    )
                                    for metric in metric_history:
                                        mlflow.log_metric(
                                            key, 
                                            metric.value, 
                                            step=metric.step,
                                            timestamp=metric.timestamp
                                        )
                                
                                # Copy tags (excluding system tags)
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