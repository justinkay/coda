import mlflow
import pandas as pd
import sqlite3
from pathlib import Path
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
    # drop seed
    parts = run_name.split("-")
    if len(parts) >= 2 and parts[-1].isdigit():
        parts = parts[:-1]
    return "-".join(parts[1:]) if len(parts) > 1 else run_name

metric = 'cumulative regret'
metric_name = metric.replace(" ", "_")
step = 100
SQL = f"""
SELECT  e.name                        AS task,
        rn.value                      AS run_name,
        m.value                       AS {metric_name},
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

WHERE   m.key  = '{metric}'
  AND   m.step = {step}               -- just step 100
  AND   r.lifecycle_stage = 'active'
  AND   e.lifecycle_stage = 'active'
"""

with sqlite3.connect(str(db_path)) as conn:
    df = pd.read_sql_query(SQL, conn)

df["method"] = df["run_name"].apply(extract_method_from_run_name)
mean_per_step = df.groupby(["task", "method", "step"], as_index=False)[metric_name].mean()
std_per_step  = df.groupby(["task", "method", "step"], as_index=False)[metric_name].std()

coda_name = 'coda-lr=0.01-mult=2.0-no-prefilter'
mean_per_step = mean_per_step[(~mean_per_step.method.str.contains('coda')) | (mean_per_step.method == coda_name)]
mean_per_step.loc[mean_per_step.method == 'activetesting', 'method'] = 'Active Testing'
mean_per_step.loc[mean_per_step.method == 'iid', 'method'] = 'Random Sampling'
mean_per_step.loc[mean_per_step.method == 'model_picker', 'method'] = 'Model Selector'
mean_per_step.loc[mean_per_step.method == 'uncertainty', 'method'] = 'Uncertainty'
mean_per_step.loc[mean_per_step.method == 'vma', 'method'] = 'VMA'
mean_per_step.loc[mean_per_step.method == coda_name, 'method'] = 'CODA (Ours)'
mean_per_step[metric_name] *= 100

std_per_step = std_per_step[(~std_per_step.method.str.contains('coda')) | (std_per_step.method == coda_name)]
std_per_step.loc[std_per_step.method == 'activetesting', 'method'] = 'Active Testing'
std_per_step.loc[std_per_step.method == 'iid', 'method'] = 'Random Sampling'
std_per_step.loc[std_per_step.method == 'model_picker', 'method'] = 'Model Selector'
std_per_step.loc[std_per_step.method == 'uncertainty', 'method'] = 'Uncertainty'
std_per_step.loc[std_per_step.method == 'vma', 'method'] = 'VMA'
std_per_step.loc[std_per_step.method == coda_name, 'method'] = 'CODA (Ours)'
std_per_step[metric_name] *= 100



methods = [ 'Random Sampling', 'Uncertainty', 'Active Testing', 'VMA', 'Model Selector', 'CODA (Ours)' ]
tasks = [
    'real_sketch',
     'real_painting',
      'real_clipart',
       'sketch_real',
        'sketch_painting', 'sketch_clipart', 'painting_real', 'painting_sketch', 'painting_clipart', 'clipart_real', 
         'clipart_sketch','clipart_painting', 'iwildcam', 'camelyon', 'fmow', 'civilcomments', 'cifar10_4070', 'cifar10_5592', 'pacs',
          'glue_cola', 'glue_mnli', 'glue_qnli', 'glue_qqp', 'glue_rte', 'glue_sst2'
]

all_vals = np.ones( shape=(len(methods), len(tasks)) ) * 999
all_stds = np.ones( shape=(len(methods), len(tasks)) ) * 999

all_vals = (
    mean_per_step
      .pivot(index="method", columns="task", values=metric_name)
      .reindex(index=methods, columns=tasks)
      .to_numpy()
)
all_stds = (
    std_per_step
      .pivot(index="method", columns="task", values=metric_name)
      .reindex(index=methods, columns=tasks)
      .to_numpy()
)

best = np.argmin(all_vals, axis=0)
second_best = np.argpartition(all_vals, 1, axis=0)[1]
best, second_best

groups = {
    "DomainNet126": [
        'real_sketch','real_painting','real_clipart',
        'sketch_real','sketch_painting','sketch_clipart',
        'painting_real','painting_sketch','painting_clipart',
        'clipart_real','clipart_sketch','clipart_painting'
    ],
    "WILDS": ['iwildcam', 'camelyon', 'fmow', 'civilcomments'],
    "MSV": ['cifar10_4070', 'cifar10_5592', 'pacs'],
    "GLUE": ['glue/cola', 'glue/mnli', 'glue/qnli', 'glue/qqp', 'glue/rte', 'glue/sst2'],
}

def pretty_task(t):
    # domain->domain pairs
    if '_' in t and not t.startswith('glue'):
        src, tgt = t.split('_', 1)
        return f"{src}$\\rightarrow${tgt}"
    if t.startswith('glue/'):
        return t.split('/', 1)[1]  # cola, mnli, ...
    if t == 'cifar10_4070':
        return 'cifar10-low'
    if t == 'cifar10_5592':
        return 'cifar10-high'
    return t

# Split method names into (row1, row2) for the header
def split_method_header(name):
    if name.startswith('CODA'):
        return (r'\cellcolor{gray!15}\textbf{CODA}', r'{\cellcolor{gray!15}\textbf{(Ours)}}')
    parts = name.split(' ', 1)
    if len(parts) == 1:
        return (parts[0], '')
    return (parts[0], parts[1])

colspec = 'cl' + 'r' * len(methods)
first_row_methods = []
second_row_methods = []
for m in methods:
    r1, r2 = split_method_header(m)
    if r2:  # two lines
        first_row_methods.append(r1)
        second_row_methods.append(r2)
    else: # one line
        first_row_methods.append(rf'\multirow{{2}}{{*}}{{{r1}}}')
        second_row_methods.append('')  # keep alignment

latex_lines = []
latex_lines.append(r'\begin{tabular}{' + colspec + '}')
latex_lines.append(r'\toprule')
latex_lines.append('')
# First header line
header_line_1 = '& \\multirow{2}{*}{Task} & ' + ' & '.join(first_row_methods) + r' \\'
latex_lines.append(header_line_1)
# Second header line
header_line_2 = '& & ' + ' & '.join([s if s else '' for s in second_row_methods]) + r'\\'
latex_lines.append(header_line_2)
latex_lines.append(r'\midrule')
latex_lines.append('')

task_to_idx = {t: j for j, t in enumerate(tasks)}
for g_name, g_tasks in groups.items():
    n_rows = len(g_tasks)
    # Parbox + rotate only for the first row of each group
    group_label = rf'\parbox[t]{{}}{{\multirow{{{n_rows}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{g_name}}}}}}}'
    for r_i, t in enumerate(g_tasks):
        task_label = pretty_task(t)
        row_cells = []
        # Build each method value
        col_j = task_to_idx[t.replace("/","_")]
        for i in range(len(methods)):
            val = all_vals[i, col_j]
            # Format value to one decimal
            val_str = f"{val:.1f}"
            # Highlighting
            if best[col_j] == i:
                val_str = rf'\textbf{{{val_str}}}'
            elif second_best[col_j] == i:
                val_str = rf'\underline{{{val_str}}}'
            if methods[i].startswith('CODA'):
                val_str = rf'\cellcolor{{gray!15}}{val_str}'
            row_cells.append(val_str)

        # Prepend group label or a blank cell if not first row
        if r_i == 0:
            row_start = f"{group_label} & {task_label} & "
        else:
            row_start = f"& {task_label} & "

        latex_lines.append(row_start + ' & '.join(row_cells) + r' \\ ')
    latex_lines.append(r'\midrule')

# Replace last \midrule by \bottomrule
latex_lines[-1] = r'\bottomrule'
latex_lines.append(r'\end{tabular}')

latex_str = "\n".join(latex_lines)
print(latex_str)
