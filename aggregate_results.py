# aggregate.py
"""Aggregate child-run metrics into their parent runs.

For every experiment and each *parent* run (a run without the
``mlflow.parentRunId`` tag), compute the step-wise mean of the chosen
metrics across all of its child runs and write those means back onto
the parent run (metric name prefixed with ``mean_``).

Usage
-----
```
python aggregate.py                # aggregates "regret" & "cumulative regret"
python aggregate.py metric1 metric2  # custom metric list
```
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

import mlflow
import numpy as np


def aggregate_metrics(metric_keys: Optional[List[str]] = None) -> None:
    """Compute step-wise means for *metric_keys* and log them to parent runs."""

    if metric_keys is None:
        metric_keys = ["regret", "cumulative regret"]

    client = mlflow.tracking.MlflowClient()

    # Iterate over every experiment visible to the current tracking URI
    for exp in client.search_experiments():
        exp_id = exp.experiment_id

        # Pull *all* runs in this experiment; we'll identify parents locally
        all_runs = mlflow.search_runs(
            experiment_ids=[exp_id], output_format="pandas"
        )
        if all_runs.empty:
            continue

        # Parent runs have *no* parentRunId tag (NaN in the dataframe)
        parent_mask = all_runs["tags.mlflow.parentRunId"].isna()
        parent_runs = all_runs[parent_mask]

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
                    mean_val = float(np.mean(values))
                    client.log_metric(
                        parent_run_id,
                        key=f"mean_{metric}",
                        value=mean_val,
                        step=step,
                    )
                    print(
                        f"[Exp {exp.name}] parent {parent_run_id[:8]} | "
                        f"step {step} mean_{metric} = {mean_val:.6f}"
                    )


def main() -> None:
    metrics = sys.argv[1:] if len(sys.argv) > 1 else None
    aggregate_metrics(metrics)


if __name__ == "__main__":
    main()