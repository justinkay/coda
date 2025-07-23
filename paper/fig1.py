import mlflow
import pandas as pd
import sqlite3
from pathlib import Path
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np


# load from mlflow
USE_DB = True
if USE_DB:
    mlflow.set_tracking_uri('sqlite:///coda.sqlite')
client = mlflow.tracking.MlflowClient(tracking_uri='sqlite:///coda.sqlite')

DB_URI = "sqlite:///coda.sqlite"
db_path = Path(DB_URI.replace("sqlite:///", "", 1)).expanduser().resolve()
if not db_path.exists():
    raise FileNotFoundError(f"Tracking DB not found: {db_path}")

def extract_method_from_run_name(run_name: str) -> str:
    # drop seed from run name
    parts = run_name.split("-")
    if len(parts) >= 2 and parts[-1].isdigit():
        parts = parts[:-1]
    return "-".join(parts[1:]) if len(parts) > 1 else run_name

SQL = """
SELECT  e.name                        AS task,
        rn.value                      AS run_name,
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
mean_per_step = df.groupby(["task", "method", "step"], as_index=False)["regret"].mean()

coda_name = 'coda-lr=0.01-mult=2.0-no-prefilter'
mean_per_step = mean_per_step[(~mean_per_step.method.str.contains('coda')) | (mean_per_step.method == coda_name)]
mean_per_step.loc[mean_per_step.method == 'activetesting', 'method'] = 'Active Testing'
mean_per_step.loc[mean_per_step.method == 'iid', 'method'] = 'Random Sampling'
mean_per_step.loc[mean_per_step.method == 'model_picker', 'method'] = 'ModelSelector'
mean_per_step.loc[mean_per_step.method == 'uncertainty', 'method'] = 'Uncertainty'
mean_per_step.loc[mean_per_step.method == 'vma', 'method'] = 'VMA'
mean_per_step.loc[mean_per_step.method == coda_name, 'method'] = 'CODA (Ours)'
mean_per_step.regret *= 100

tasks = mean_per_step.task.unique()
global_methods = [ 'Random Sampling', 'Uncertainty', 'Active Testing', 'VMA', 'ModelSelector', 'CODA (Ours)' ]
steps_until_below_1 = {task: {} for task in tasks}
palette = sns.color_palette("colorblind", n_colors=len(global_methods))
color_dict = {raw: color for raw, color in zip(global_methods, palette[::-1])}

## plot
THRESHOLD = 1.0        # 1%
N_CONSECUTIVE = 1000   # must have < THRESHOLD for these many consecutive steps
MAX_STEPS = 100
NO_CONVERGENCE = 999

# optional: sliding window average regrets 
# disabled by default (window size = 1)
WINDOW_SIZE = 1
rolling_regrets = {task: {} for task in tasks}
for task in tasks:
    for method in global_methods:
        raw_regrets = mean_per_step[(mean_per_step.task == task) & (mean_per_step.method == method)]
        rolled = raw_regrets.sort_values(by='step').regret.rolling(
            window=WINDOW_SIZE, center=True, min_periods=1
        ).mean()
        rolling_regrets[task][method] = list(rolled)

# get convergence time for each method/task combo
convergence_time = {method: {} for method in global_methods}
for method in global_methods:
    for task in tasks:
        converge_step = NO_CONVERGENCE
        for start_idx in range(MAX_STEPS):
            window_vals = rolling_regrets[task][method][start_idx:]
            if all(val < THRESHOLD for val in window_vals):
                converge_step = start_idx + 1 # start at step 1
                break  # no need to keep searching

        convergence_time[method][task] = converge_step

# get proportion of tasks converged at each step for each method
method_proportions = {method: np.zeros(MAX_STEPS) for method in global_methods}
for method in global_methods:
    # For each step from 1..MAX_STEPS, count how many tasks are "converged" by that step
    for s in range(1, MAX_STEPS + 1):
        count_converged = sum(
            1 for task in tasks 
            if convergence_time[method][task] != NO_CONVERGENCE and 
               convergence_time[method][task] <= s
        )
        method_proportions[method][s-1] = count_converged / len(tasks)

# plot
fig, ax = plt.subplots(figsize=(5.5,5))
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams["font.serif"] = ["Nimbus Roman"]

PLOT_WINDOW_SIZE=10 # for smoothing the plot
for method in global_methods:
    yvals = method_proportions[method]
    ax.plot(
        range(1, MAX_STEPS+1),
        pd.Series(yvals).rolling(window=PLOT_WINDOW_SIZE, min_periods=None, center=True).mean(),
        label=method,
        color=color_dict[method],
        linewidth=3.5 if "Ours" in method else 2,
        alpha=1.0 if "Ours" in method else 0.5,
    )

ax.set_xlabel("Number of labels queried", fontsize=19)
ax.set_ylabel(f"Fraction of benchmarks\nconverged to < 1% regret", fontsize=19)
ax.set_title("Model Selection Label Efficiency", fontsize=19, y=1.01)
ax.grid(True,which='both',alpha=0.5)

# legend
bold = {'weight':'bold'}
legend_handles = []
for method in global_methods: #[:-1]:
    handle = mlines.Line2D(
        [], [],
        color=color_dict[method],
        linewidth=4 if "Ours" in method else 3,
        label=method, #.replace(' ', '\n'),
        alpha=1.0 if "Ours" in method else 0.5,
    )
    legend_handles.append(handle)
ax.legend(handles=legend_handles, #[:-1], 
           loc="upper center", ncol=2, 
           labelspacing=0.15,
           columnspacing=1,
           title_fontsize=16,
           bbox_to_anchor=(0.5,1.01))

ax.set_ylim(-0.05, 1.05)
plt.savefig('fig1_cameraready.pdf', bbox_inches='tight')