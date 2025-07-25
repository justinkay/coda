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


task_groups = [ 

    # row 1
    ['painting_real',
    'painting_sketch',
    'painting_clipart',

    'sketch_painting',
    'sketch_real',
    'sketch_clipart'],

    # row 2
    ['clipart_real',
    'clipart_sketch',
    'clipart_painting',

    'real_painting',
    'real_sketch',
    'real_clipart'],

    # row 3
    ["iwildcam", 
     "fmow",
    "civilcomments",
    "camelyon",
    "cifar10_4070",
    "cifar10_5592",
    "pacs"],

    # row 4
    ["glue/cola",
    "glue/mnli",
    "glue/qnli",
    "glue/qqp",
     "glue/rte", 
     "glue/sst2",
     "glue/mrpc"], 

]


import numpy as np

global_methods = [ 'Random Sampling', 'Uncertainty', 'Active Testing', 'VMA', 'ModelSelector', 'CODA (Ours)' ]
baselines = global_methods
task_to_name_to_reg_means = {}
task_to_name_to_creg_means = {}

for k, task_group in enumerate(task_groups):
    for j, task in enumerate(task_group):
        task_to_name_to_reg_means[task] = {}
        task_to_name_to_creg_means[task] = {}
        for i, nice_name in enumerate(baselines):
            raw_regrets = regret_mean_per_step[(regret_mean_per_step.task == task.replace("/","_")) & (regret_mean_per_step.method == nice_name)]
            WINDOW_SIZE = 1
            rolled = raw_regrets.sort_values(by='step').regret.rolling(
                window=WINDOW_SIZE, center=True, min_periods=1
            ).mean()
            if not len(rolled.values):
                print("PROBLEM", task, nice_name)
            task_to_name_to_reg_means[task][nice_name] = np.array(rolled.values)
            
            raw_cuml_regrets = cuml_regret_mean_per_step[(cuml_regret_mean_per_step.task == task.replace("/","_")) & (cuml_regret_mean_per_step.method == nice_name)]
            WINDOW_SIZE = 1
            rolled = raw_cuml_regrets.sort_values(by='step').cuml_regret.rolling(
                window=WINDOW_SIZE, center=True, min_periods=1
            ).mean()
            task_to_name_to_creg_means[task][nice_name] = np.array(rolled.values)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib as mpl
import matplotlib.ticker as ticker


metrics=["Regret (loss)", "Cumulative regret (loss)"]

plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams["font.serif"] = ["Nimbus Roman"]

palette = sns.color_palette("colorblind", n_colors=len(global_methods))
color_dict = {raw: color for raw, color in zip(global_methods, palette[::-1])}

lss = [] # line styles

for k, tasks in enumerate(task_groups):

    fig, axes = plt.subplots(2, len(tasks), figsize=(3 * len(tasks), 6)) #, sharex='col')

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

                # print(method)
                ax.plot(list(range(1,101)), mean_vals, label=method, color=color_dict[method],
                        linestyle=ls, marker=marker, markevery=30, 
                        linewidth=2 if method == 'CODA (Ours)' else 1.5, 
                        alpha=1.0 if method == 'CODA (Ours)' else 0.8)
            
            if row == 1 and col == 0:
                ax.set_xlabel("Steps", fontsize=18, x=3.5)
            if col == 0:
                ax.set_ylabel(metric.replace(" (loss)", "").replace("Cumulative", "Cuml."), fontsize=18)

            ax.grid(True)
            if row == 0:
                ax.set_title(f"{task}".replace("_", "→"), fontsize=18)


    handles = []
    for method in global_methods:
        ls = 'solid'
        marker = None
        handle = mlines.Line2D([], [], color=color_dict[method], label=method,
                                linestyle=ls, linewidth=3.0, marker=marker, markersize=5)
        handles.append(handle)


    if k == 0:
        leg = fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15), fontsize=22)

    plt.subplots_adjust(wspace=0.23) 
    plt.savefig(f'fig5-{k}-cameraready.pdf',dpi=300, bbox_inches="tight")
    plt.clf() # clear
