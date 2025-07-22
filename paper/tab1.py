import mlflow
import pandas as pd
import numpy as np
import os

USE_DB = True
if USE_DB:
    mlflow.set_tracking_uri('sqlite:///coda.sqlite')


def get_baseline_methods():
    return [
        ('iid', 'Random Sampling'),
        ('uncertainty', 'Uncertainty'),
        ('activetesting', 'Active Testing'),
        ('vma', 'VMA'),
        ('model_picker', 'Model Selector'),
        # ('coda', 'CODA'),

        ('coda-lr=0.01-mult=1.0', 'coda-lr=0.01-mult=1.0'),
        ('coda-lr=0.01-mult=2.0', 'coda-lr=0.01-mult=2.0'),
        ('coda-lr=0.01-mult=5.0', 'coda-lr=0.01-mult=5.0'),
        ('coda-lr=0.01-mult=10.0', 'coda-lr=0.01-mult=10.0'),

        ('coda-lr=0.001-mult=1.0', 'coda-lr=0.001-mult=1.0'),
        ('coda-lr=0.001-mult=2.0', 'coda-lr=0.001-mult=2.0'),
        ('coda-lr=0.001-mult=5.0', 'coda-lr=0.001-mult=5.0'),
    ]

def normalize_task_name(experiment_name):
    if experiment_name.startswith('glue_'):
        return experiment_name.replace('_', '/')
    return experiment_name

def extract_method_from_run_name(run_name):
    parts = run_name.split('-')
    if len(parts) >= 2:
        # Remove task (first part) and seed (last part if it's a digit)  
        if parts[-1].isdigit():
            return '-'.join(parts[1:-1])
        else:
            return '-'.join(parts[1:])
    return run_name

def load_all_baseline_results():
    client = mlflow.tracking.MlflowClient()
    
    tasks = [
        'real_sketch', 'real_painting', 'real_clipart',
        'sketch_real', 'sketch_painting', 'sketch_clipart', 
        'painting_real', 'painting_sketch', 'painting_clipart', 
        'clipart_real', 'clipart_sketch', 'clipart_painting', 
        'iwildcam', 'camelyon', 'fmow', 'civilcomments', 
        'cifar10_4070', 'cifar10_5592', 'pacs',
        'glue_cola', 'glue_mnli', 'glue_qnli', 'glue_qqp', 'glue_rte', 'glue_sst2'
    ]
    
    all_experiments = client.search_experiments()
    experiments_by_name = {}
    
    for exp in all_experiments:
        if exp.name in tasks:
            if exp.name not in experiments_by_name:
                experiments_by_name[exp.name] = []
            experiments_by_name[exp.name].append(exp)
    
    print(f"Found experiments for {len(experiments_by_name)} tasks")
    
    all_data = []
    
    for task_name in tasks:
        if task_name not in experiments_by_name:
            print(f"Warning: No experiments found for {task_name}")
            continue
            
        print(f"Processing {task_name}...")
        
        for exp in experiments_by_name[task_name]:
            runs = client.search_runs([exp.experiment_id], max_results=500)
            
            # Get child runs (they have detailed step data)
            child_runs = [r for r in runs if 'mlflow.parentRunId' in r.data.tags]
            
            for run in child_runs:
                run_name = run.info.run_name
                
                # TESTING
                # Skip CODA runs
                # if 'coda' in run_name.lower():
                #     continue
                
                method = extract_method_from_run_name(run_name)
                seed = int(run.data.params.get('seed', 0))
                
                try:
                    # Get cumulative regret at step 100
                    cum_regret_history = client.get_metric_history(run.info.run_id, "cumulative regret")
                    
                    step_100_cum_regret = None
                    for metric in cum_regret_history:
                        if metric.step == 100:
                            step_100_cum_regret = metric.value
                            break
                    
                    if step_100_cum_regret is not None:
                        normalized_task = normalize_task_name(task_name)
                        dataset = task_name.split('_')[0] if '_' in task_name else 'unknown'
                        
                        row = {
                            '_step': 100,
                            'Cumulative regret (loss)': step_100_cum_regret,
                            'alg-detail': method,
                            'seed': seed,
                            'dataset': dataset, 
                            'task': normalized_task
                        }
                        all_data.append(row)
                
                except Exception as e:
                    print(f"  Error processing {run_name}: {e}")
                    continue
    
    return pd.DataFrame(all_data)

def generate_full_latex_table(df):
    baselines = get_baseline_methods()
    
    tasks = [
        'real_sketch', 'real_painting', 'real_clipart',
        'sketch_real', 'sketch_painting', 'sketch_clipart', 
        'painting_real', 'painting_sketch', 'painting_clipart', 
        'clipart_real', 'clipart_sketch', 'clipart_painting', 
        'iwildcam', 'camelyon', 'fmow', 'civilcomments', 
        'cifar10_4070', 'cifar10_5592', 'pacs',
        'glue/cola', 'glue/mnli', 'glue/qnli', 'glue/qqp', 'glue/rte', 'glue/sst2'
    ]
    
    num_methods = len(baselines)
    num_tasks = len(tasks)
    
    all_vals = np.ones((num_methods, num_tasks)) * 999
    all_stds = np.ones((num_methods, num_tasks)) * 999
    
    # baseline results
    for i, (alg, nice_name) in enumerate(baselines):
        for j, task in enumerate(tasks):
            task_data = df[(df['task'] == task) & (df['alg-detail'] == alg) & (df['_step'] == 100)]
            
            if len(task_data) > 0:
                mean_val = task_data['Cumulative regret (loss)'].mean() * 100
                std_val = task_data['Cumulative regret (loss)'].std() * 100
                all_vals[i, j] = mean_val
                all_stds[i, j] = std_val if not np.isnan(std_val) else 0
    
    all_vals = np.nan_to_num(all_vals, nan=999)
    all_stds = np.nan_to_num(all_stds, nan=999)
    
    # Find best and second best
    best = np.argmin(all_vals, axis=0)
    second_best = np.argpartition(all_vals, 1, axis=0)[1]
    
    latex_str = ''

    # for every method (rows)
    for i in range(num_methods):
        
        # get method name
        name = baselines[i][1]

        # to save width, methods with 2-word names get 2 rows
        split = name.split(' ', maxsplit=1)
        multi_name = False
        if len(split) == 2:
            multi_name = True
            name_part_1 = split[0]
            name_part_2 = split[1]
            latex_str += name_part_1 + ' & '
        else:
            latex_str += name + ' & '
        
        # for every task (columns)
        for j in range(len(all_vals[0])):
            opened = False

            if multi_name:
                latex_str += '\\multirow{2}{*}{ '

            if best[j] == i:
                latex_str += '\\textbf{'
                opened = True
            elif second_best[j] == i:
                latex_str += '\\underline{'
                opened = True
            
            latex_str += "{:.1f}".format(all_vals[i,j]) 
            
            if opened:
                latex_str += "}"
            
            if multi_name:
                latex_str += '}'

            latex_str += "{\\small $\\pm$ " + "{:.1f}".format(all_stds[i,j]) + "}"

            if j != len(all_vals[0]) - 1: 
                latex_str += ' & '
        
        latex_str += '\\\\ \n'

        # multi-row second line
        if multi_name:
            latex_str += f'{name_part_2} & & & & & & & & & & & & & & & & & & & & & & & & & \\\\ \n'
    
    return latex_str

if __name__ == "__main__":
    print("Loading all baseline data from MLFlow...")
    
    # Load the data
    df = load_all_baseline_results()
    
    if len(df) == 0:
        print("No data loaded. Check MLFlow experiments.")
        exit(1)
    
    print(f"\nLoaded {len(df)} data points")
    print(f"Tasks found: {len(df['task'].unique())}")
    print(f"Methods found: {sorted(df['alg-detail'].unique())}")
    
    # Show data coverage
    coverage = df.groupby(['task', 'alg-detail']).size().reset_index(name='seeds')
    print(f"\nData coverage:")
    for task in sorted(df['task'].unique()):
        task_coverage = coverage[coverage['task'] == task]
        methods = task_coverage.set_index('alg-detail')['seeds'].to_dict()
        print(f"  {task}: {methods}")
    
    # Generate full table
    print("\n" + "="*80)
    print("Generating complete LaTeX table...")
    latex_table = generate_full_latex_table(df)
    
    print("LATEX TABLE:")
    print("="*80)
    print(latex_table)
    
    # Save to file
    output_file = 'tab1.tex'
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"\nComplete table saved to {output_file}")