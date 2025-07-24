import mlflow
import pandas as pd
import sqlite3
from pathlib import Path

# load from mlflow
USE_DB = True
if USE_DB:
    mlflow.set_tracking_uri('sqlite:///coda.sqlite')
client = mlflow.tracking.MlflowClient(tracking_uri='sqlite:///coda.sqlite')

DB_URI = "sqlite:///coda.sqlite"          # same URI you pass to mlflow
db_path = Path(DB_URI.replace("sqlite:///", "", 1)).expanduser().resolve()
if not db_path.exists():
    raise FileNotFoundError(f"Tracking DB not found: {db_path}")

def extract_method_from_run_name(run_name: str) -> str:
    parts = run_name.split("-")
    if len(parts) >= 2 and parts[-1].isdigit():       # last part looks like a seed
        parts = parts[:-1]                            # drop seed
    return "-".join(parts[1:]) if len(parts) > 1 else run_name

SQL = """
SELECT  e.name                        AS task,
        rn.value                      AS run_name,   --  run name comes from tag
        m.value                       AS regret,
        m.step                        AS step
FROM    metrics   m
JOIN    runs      r   ON m.run_uuid      = r.run_uuid
JOIN    experiments e ON r.experiment_id = e.experiment_id

-- keep **child** runs only
JOIN    tags t_parent
       ON r.run_uuid = t_parent.run_uuid
      AND t_parent.key = 'mlflow.parentRunId'

-- run-name tag: key = 'mlflow.runName'
LEFT JOIN tags rn
       ON r.run_uuid = rn.run_uuid
      AND rn.key     = 'mlflow.runName'

WHERE   m.key  = 'regret'
  AND   r.lifecycle_stage = 'active'
  AND   e.lifecycle_stage = 'active'
"""

with sqlite3.connect(str(db_path)) as conn:
    df = pd.read_sql_query(SQL, conn)

df["method"] = df["run_name"].apply(extract_method_from_run_name)
regret_mean_per_step = df.groupby(["task", "method", "step"], as_index=False)["regret"].mean()

coda_name = 'coda-lr=0.01-mult=2.0-no-prefilter'
regret_mean_per_step = regret_mean_per_step[(~regret_mean_per_step.method.str.contains('coda')) | (regret_mean_per_step.method == coda_name)]
regret_mean_per_step.loc[regret_mean_per_step.method == 'activetesting', 'method'] = 'Active Testing'
regret_mean_per_step.loc[regret_mean_per_step.method == 'iid', 'method'] = 'Random Sampling'
regret_mean_per_step.loc[regret_mean_per_step.method == 'model_picker', 'method'] = 'ModelSelector'
regret_mean_per_step.loc[regret_mean_per_step.method == 'uncertainty', 'method'] = 'Uncertainty'
regret_mean_per_step.loc[regret_mean_per_step.method == 'vma', 'method'] = 'VMA'
regret_mean_per_step.loc[regret_mean_per_step.method == coda_name, 'method'] = 'CODA (Ours)'
regret_mean_per_step.regret *= 100

regret_mean_per_step.head()

import mlflow
import pandas as pd
import sqlite3
from pathlib import Path

# load from mlflow
USE_DB = True
if USE_DB:
    mlflow.set_tracking_uri('sqlite:///coda.sqlite')
client = mlflow.tracking.MlflowClient(tracking_uri='sqlite:///coda.sqlite')

DB_URI = "sqlite:///coda.sqlite"          # same URI you pass to mlflow
db_path = Path(DB_URI.replace("sqlite:///", "", 1)).expanduser().resolve()
if not db_path.exists():
    raise FileNotFoundError(f"Tracking DB not found: {db_path}")

def extract_method_from_run_name(run_name: str) -> str:
    parts = run_name.split("-")
    if len(parts) >= 2 and parts[-1].isdigit():       # last part looks like a seed
        parts = parts[:-1]                            # drop seed
    return "-".join(parts[1:]) if len(parts) > 1 else run_name

SQL = """
SELECT  e.name                        AS task,
        rn.value                      AS run_name,   --  run name comes from tag
        m.value                       AS cuml_regret,
        m.step                        AS step
FROM    metrics   m
JOIN    runs      r   ON m.run_uuid      = r.run_uuid
JOIN    experiments e ON r.experiment_id = e.experiment_id

-- keep **child** runs only
JOIN    tags t_parent
       ON r.run_uuid = t_parent.run_uuid
      AND t_parent.key = 'mlflow.parentRunId'

-- run-name tag: key = 'mlflow.runName'
LEFT JOIN tags rn
       ON r.run_uuid = rn.run_uuid
      AND rn.key     = 'mlflow.runName'

WHERE   m.key  = 'cumulative regret'
  AND   r.lifecycle_stage = 'active'
  AND   e.lifecycle_stage = 'active'
"""

with sqlite3.connect(str(db_path)) as conn:
    df = pd.read_sql_query(SQL, conn)

df["method"] = df["run_name"].apply(extract_method_from_run_name)
cuml_regret_mean_per_step = df.groupby(["task", "method", "step"], as_index=False)["cuml_regret"].mean()

coda_name = 'coda-lr=0.01-mult=2.0-no-prefilter'
cuml_regret_mean_per_step = cuml_regret_mean_per_step[(~cuml_regret_mean_per_step.method.str.contains('coda')) | (cuml_regret_mean_per_step.method == coda_name)]
cuml_regret_mean_per_step.loc[cuml_regret_mean_per_step.method == 'activetesting', 'method'] = 'Active Testing'
cuml_regret_mean_per_step.loc[cuml_regret_mean_per_step.method == 'iid', 'method'] = 'Random Sampling'
cuml_regret_mean_per_step.loc[cuml_regret_mean_per_step.method == 'model_picker', 'method'] = 'ModelSelector'
cuml_regret_mean_per_step.loc[cuml_regret_mean_per_step.method == 'uncertainty', 'method'] = 'Uncertainty'
cuml_regret_mean_per_step.loc[cuml_regret_mean_per_step.method == 'vma', 'method'] = 'VMA'
cuml_regret_mean_per_step.loc[cuml_regret_mean_per_step.method == coda_name, 'method'] = 'CODA (Ours)'
cuml_regret_mean_per_step.cuml_regret *= 100

cuml_regret_mean_per_step.head()

memory_use_gb = {
      
                'MSV\n7-10 class': {
                    'cifar10_4070': 0.04063744, 
                    'cifar10_5592': 0.04063744, 
                    'pacs': 0.016964096,
                },
                 
                # ModelSelector
                'GLUE\n2-3 class': {
                    'glue/cola': 0.009445376, 
                    'glue/mnli': 0.018265088,
                    'glue/qnli': 0.012504064, 
                    'glue/qqp': 0.042404864, 
                    'glue/rte': 0.00872192, 
                    'glue/sst2': 0.00921088,
                    'glue/mrpc': 0.008840192,  

                    # not included - fewer than 100 data points
                    # 'glue/wnli': 0.008594944
                 },

                
                # WILDS
                'WILDS Multiclass\n62-182 class': {
                    # 'civilcomments': 0.031593984, 
                    'fmow': 1.32826112, 
                    'iwildcam': 1.510516736, 
                    # 'camelyon': 0.036469248,
                },

                'WILDS Binary\n2-class ': {
                    'civilcomments': 0.031593984, 
                    # 'fmow': 1.32826112, 
                    # 'iwildcam': 1.510516736, 
                    'camelyon': 0.036469248,
                },

                # all wilds together
                # 'WILDS\n2-182 class:': {
                #     'civilcomments': 0.031593984, 
                #     'fmow': 1.32826112, 
                #     'iwildcam': 1.510516736, 
                #     'camelyon': 0.036469248,
                # },


                # DomainNet 
                'DomainNet\n126-class': {
                    'real_sketch': 3.758885376, 
                    'real_clipart': 2.900022784, 
                    'real_painting': 1.628145152, 
                    'sketch_real': 9.98845184, 
                    'sketch_clipart': 2.900022784, 
                    'sketch_painting': 1.628145152, 
                    'clipart_real': 6.378751488, 
                    'clipart_sketch': 3.232947712,
                    'clipart_painting': 1.628145152, 
                    'painting_real': 9.98845184, 
                    'painting_sketch': 3.157962752, 
                    'painting_clipart': 2.900022784,
                },

                
            }

import numpy as np

global_methods = [ 'Random Sampling', 'Uncertainty', 'Active Testing', 'VMA', 'ModelSelector', 'CODA (Ours)' ]
baselines = global_methods
task_to_name_to_reg_means = {}
task_to_name_to_creg_means = {}

MEDIAN = True

for dataset in memory_use_gb.keys():

    task_to_name_to_reg_means[dataset] = {}
    task_to_name_to_creg_means[dataset] = {}

    for i, nice_name in enumerate(baselines):

        # average over all tasks in dataset
        averaged_regs = np.zeros( (len(memory_use_gb[dataset]),100 ))
        averaged_cuml_regs = np.zeros( (len(memory_use_gb[dataset]),100))
        
        for j, task in enumerate(memory_use_gb[dataset].keys()):
            raw_regrets = regret_mean_per_step[(regret_mean_per_step.task == task.replace("/","_")) & (regret_mean_per_step.method == nice_name)]
            WINDOW_SIZE = 1
            rolled = raw_regrets.sort_values(by='step').regret.rolling(
                window=WINDOW_SIZE, center=True, min_periods=1
            ).mean()
            if not len(rolled.values):
                print("PROBLEM", task, nice_name)
            averaged_regs[j] = rolled.values
            
            raw_cuml_regrets = cuml_regret_mean_per_step[(cuml_regret_mean_per_step.task == task.replace("/","_")) & (cuml_regret_mean_per_step.method == nice_name)]
            WINDOW_SIZE = 1
            rolled = raw_cuml_regrets.sort_values(by='step').cuml_regret.rolling(
                window=WINDOW_SIZE, center=True, min_periods=1
            ).mean()
            averaged_cuml_regs[j] = np.array(rolled.values)

        if MEDIAN:
            averaged_regs = np.median(averaged_regs, axis=0)
            averaged_cuml_regs = np.median(averaged_cuml_regs, axis=0)
        else:
            averaged_regs = averaged_regs.mean(axis=0)
            averaged_cuml_regs = averaged_cuml_regs.mean(axis=0)

        task_to_name_to_reg_means[dataset][nice_name] = averaged_regs
        task_to_name_to_creg_means[dataset][nice_name] = averaged_cuml_regs

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines

metrics=["Regret (loss)", "Cumulative regret (loss)"]

plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams["font.serif"] = ["Nimbus Roman"]

global_methods = [ 'Random Sampling', 'Uncertainty', 'Active Testing', 'VMA', 'ModelSelector', 'CODA (Ours)' ]
palette = sns.color_palette("colorblind", n_colors=len(global_methods))
color_dict = {raw: color for raw, color in zip(global_methods, palette[::-1])}

lss = [] # line styles

tasks = list(memory_use_gb.keys())
num_tasks = len(tasks)

fig, axes = plt.subplots(2, num_tasks, figsize=(3 * num_tasks, 6)) #, sharex='col')

for col, task in enumerate(tasks):

    for row, metric in enumerate(metrics):
        ax = axes[row, col]

        if metric == 'Regret (loss)':
            means = task_to_name_to_reg_means[task]
        else:
            means = task_to_name_to_creg_means[task]

        for method in global_methods:
            ls = 'solid'
            marker = None
            lss.append(ls)
            mean_vals = means[method]

            if metric == "Regret (loss)":
                window_size = 10
                mean_vals = pd.Series(mean_vals).rolling(window=window_size, center=True, min_periods=1).mean()
            else:
                window_size = 5
                mean_vals = pd.Series(mean_vals).rolling(window=window_size, center=True, min_periods=1).mean()

            print(method)
            ax.plot(list(range(1,101)), mean_vals, label=method, color=color_dict[method],
                    linestyle=ls, marker=marker, markevery=30, 
                    linewidth=2 if method == 'CODA (Ours)' else 1.5, 
                    alpha=1.0 if method == 'CODA (Ours)' else 0.8)
        
        if row == 1 and col == 0:
            ax.set_xlabel("Steps", fontsize=24, x=3)
        if col == 0:
            ax.set_ylabel(metric.replace(" (loss)", "").replace("Cumulative", "Cuml."), fontsize=24)

        ax.grid(True)
        if row == 0:
            ax.set_title(f"{task}".replace("_", "→"), fontsize=24)
        

# Build global legend handles from the global_methods.
handles = []
for method in global_methods:
    ls = 'solid'
    marker = None
    handle = mlines.Line2D([], [], color=color_dict[method], label=method,
                            linestyle=ls, linewidth=3.0, marker=marker, markersize=5)
    handles.append(handle)

# Place a shared legend above all subplots in three columns.
leg = fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.23), fontsize=26)

plt.subplots_adjust(wspace=0.23) 
plt.savefig('fig3_cameraready.pdf',dpi=300, bbox_inches="tight")
