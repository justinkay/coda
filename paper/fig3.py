import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os


def get_baseline_methods():
    return [
        ('iid', 'Random Sampling'),
        ('uncertainty', 'Uncertainty'),
        ('activetesting', 'Active Testing'),
        ('vma', 'VMA'),
        ('model_picker', 'Model Selector'),
    ]

def normalize_task_name(experiment_name):
    if experiment_name.startswith('glue_'):
        return experiment_name.replace('_', '/')
    return experiment_name

def extract_method_from_run_name(run_name):
    parts = run_name.split('-')
    if len(parts) >= 2:
        if parts[-1].isdigit():
            return '-'.join(parts[1:-1])
        else:
            return '-'.join(parts[1:])
    return run_name

def get_dataset_groups():
    return {
        'MSV\n7-10 class': ['cifar10_4070', 'cifar10_5592', 'pacs'],
        'GLUE\n2-3 class': ['glue/cola', 'glue/qnli', 'glue/qqp', 'glue/rte', 'glue/sst2'],
        'WILDS Multiclass\n62-182 class': ['fmow', 'iwildcam'],
        'WILDS Binary\n2-class ': ['civilcomments', 'camelyon'],
        'DomainNet\n126-class': [
            'real_sketch', 'real_clipart', 'real_painting',
            'sketch_real', 'sketch_clipart', 'sketch_painting', 
            'clipart_real', 'clipart_sketch', 'clipart_painting',
            'painting_real', 'painting_sketch', 'painting_clipart'
        ],
    }

def load_timeseries_data_for_plotting():
    client = mlflow.tracking.MlflowClient()
    
    dataset_groups = get_dataset_groups()
    all_tasks = []
    for tasks in dataset_groups.values():
        all_tasks.extend(tasks)
    
    mlflow_tasks = []
    for task in all_tasks:
        if task.startswith('glue/'):
            mlflow_tasks.append(task.replace('/', '_'))
        else:
            mlflow_tasks.append(task)
    
    all_experiments = client.search_experiments()
    experiments_by_name = {}
    
    for exp in all_experiments:
        if exp.name in mlflow_tasks:
            if exp.name not in experiments_by_name:
                experiments_by_name[exp.name] = []
            experiments_by_name[exp.name].append(exp)
    
    print(f"Loading timeseries data for {len(experiments_by_name)} tasks...")
    
    all_data = []
    
    for task_name in mlflow_tasks:
        if task_name not in experiments_by_name:
            print(f"Warning: No experiments found for {task_name}")
            continue
            
        print(f"  Loading {task_name}...")
        
        for exp in experiments_by_name[task_name]:
            runs = client.search_runs([exp.experiment_id], max_results=500)
            child_runs = [r for r in runs if 'mlflow.parentRunId' in r.data.tags]
            
            for run in child_runs:
                run_name = run.info.run_name
                
                # TEST
                # Skip CODA runs
                if 'coda' in run_name.lower():
                    continue
                
                method = extract_method_from_run_name(run_name)
                seed = int(run.data.params.get('seed', 0))
                
                try:
                    regret_history = client.get_metric_history(run.info.run_id, "regret")
                    cum_regret_history = client.get_metric_history(run.info.run_id, "cumulative regret")
                    
                    for regret_metric, cum_regret_metric in zip(regret_history, cum_regret_history):
                        if regret_metric.step <= 100:
                            row = {
                                '_step': regret_metric.step,
                                'Regret (loss)': regret_metric.value,
                                'Cumulative regret (loss)': cum_regret_metric.value,
                                'alg-detail': method,
                                'seed': seed,
                                'task': normalize_task_name(task_name)
                            }
                            all_data.append(row)
                
                except Exception as e:
                    print(f"    Error loading {run_name}: {e}")
                    continue
    
    return pd.DataFrame(all_data)

def compute_averaged_metrics_by_dataset(df):
    baselines = get_baseline_methods()
    dataset_groups = get_dataset_groups()
    
    task_to_name_to_reg_means = {}
    task_to_name_to_creg_means = {}
    
    for dataset, tasks in dataset_groups.items():
        print(f"Computing averages for: {dataset}")
        
        task_to_name_to_reg_means[dataset] = {}
        task_to_name_to_creg_means[dataset] = {}
        
        # Process each baseline method
        for alg, nice_name in baselines:
            averaged_regs = np.zeros(100)
            averaged_cum_regs = np.zeros(100)
            valid_tasks = 0
            
            for task in tasks:
                task_df = df[df['task'] == task]
                baseline_df = task_df[task_df['alg-detail'] == alg]
                
                if len(baseline_df) == 0:
                    continue
                
                regs = []
                cregs = []
                
                for step in range(1, 101):
                    step_data = baseline_df[baseline_df['_step'] == step]
                    if len(step_data) > 0:
                        reg = step_data['Regret (loss)'].mean() * 100
                        creg = step_data['Cumulative regret (loss)'].mean() * 100
                        regs.append(reg)
                        cregs.append(creg)
                    else:
                        # Fill missing data with zeros or previous values
                        regs.append(regs[-1] if regs else 0)
                        cregs.append(cregs[-1] if cregs else 0)
                
                if len(regs) == 100:
                    averaged_regs += np.array(regs)
                    averaged_cum_regs += np.array(cregs)
                    valid_tasks += 1
            
            if valid_tasks > 0:
                averaged_regs /= valid_tasks
                averaged_cum_regs /= valid_tasks
                
                task_to_name_to_reg_means[dataset][nice_name] = averaged_regs
                task_to_name_to_creg_means[dataset][nice_name] = averaged_cum_regs
    
    return task_to_name_to_reg_means, task_to_name_to_creg_means

def create_regret_plots(reg_means, creg_means):
    baselines = get_baseline_methods()
    global_methods = [nice_name for _, nice_name in baselines]
    
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["font.serif"] = ["FreeSerif"]
    colors = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9']
    palette = colors[:len(global_methods)]
    color_dict = {method: color for method, color in zip(global_methods, palette[::-1])}
    
    metrics = ["Regret (loss)", "Cumulative regret (loss)"]
    datasets = list(reg_means.keys())
    num_datasets = len(datasets)
    
    fig, axes = plt.subplots(2, num_datasets, figsize=(3 * num_datasets, 6))
    
    # Handle single dataset case
    if num_datasets == 1:
        axes = axes.reshape(-1, 1)
    
    for col, dataset in enumerate(datasets):
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            
            if metric == 'Regret (loss)':
                means_dict = reg_means[dataset]
            else:
                means_dict = creg_means[dataset]
            
            for method in global_methods:
                if method not in means_dict:
                    continue
                    
                mean_vals = means_dict[method]
                
                # Apply smoothing 
                if metric == "Regret (loss)":
                    window_size = 10
                else:
                    window_size = 5
                
                mean_vals = pd.Series(mean_vals).rolling(window=window_size, center=True, min_periods=1).mean()
                
                ax.plot(list(range(1, 101)), mean_vals, 
                       label=method, color=color_dict[method],
                       linestyle='solid', markevery=30, markersize=5)
            
            if row == 1 and col == 0:
                ax.set_xlabel("Steps", fontsize=24, x=num_datasets/2)
            if col == 0:
                ax.set_ylabel(metric.replace(" (loss)", "").replace("Cumulative", "Cuml."), fontsize=24)
            
            ax.grid(True)
            if row == 0:
                ax.set_title(f"{dataset}".replace("_", "→"), fontsize=24)
    
    # global legend
    handles = []
    for method in global_methods:
        handle = mlines.Line2D([], [], color=color_dict[method], label=method,
                              linestyle='solid', linewidth=3.0, markersize=5)
        handles.append(handle)
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.23), fontsize=26)
    
    plt.subplots_adjust(wspace=0.23)
    
    output_file = 'fig3.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {output_file}")
    plt.savefig(output_file.replace('.pdf', '.png'), dpi=300, bbox_inches="tight")
    
    plt.show()

if __name__ == "__main__":
    print("Loading MLFlow data for plotting...")
    
    # Load timeseries data
    df = load_timeseries_data_for_plotting()
    
    if len(df) == 0:
        print("No data loaded. Check MLFlow experiments.")
        exit(1)
    
    print(f"\nLoaded {len(df)} data points")
    print(f"Tasks: {len(df['task'].unique())}")
    print(f"Methods: {sorted(df['alg-detail'].unique())}")
    print(f"Step range: {df['_step'].min()} to {df['_step'].max()}")
    
    # Compute averaged metrics
    print("\nComputing averaged metrics by dataset...")
    reg_means, creg_means = compute_averaged_metrics_by_dataset(df)
    
    print(f"\nDataset groups processed: {list(reg_means.keys())}")
    
    # Create plots
    print("\nCreating regret plots...")
    create_regret_plots(reg_means, creg_means)
    
    print("Done!")