"""Publish CODA benchmark predictions as a Hugging Face dataset.

Requires huggingface_hub (pip install huggingface_hub) and a Hugging Face login.
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import huggingface_hub
import torch


MODELSELECTOR_TASKS = frozenset("""
cifar10_4070 cifar10_5592 pacs glue_cola glue_mnli glue_qnli glue_qqp
glue_rte glue_sst2 glue_mrpc
""".split())

KNOWN_TASKS = MODELSELECTOR_TASKS | frozenset("""
sketch_painting painting_clipart real_painting clipart_painting painting_real
sketch_clipart real_clipart real_sketch painting_sketch clipart_real
clipart_sketch sketch_real iwildcam fmow camelyon civilcomments
""".split())

LICENCE = (
    "Prediction files released under CC0 1.0 at the benchmark author's choice. "
    "The underlying datasets keep their own terms: DomainNet "
    "(no licence, fair-use policy, https://ai.bu.edu/M3SDA/#refs), "
    "iWildCam (CDLA-Permissive-1.0), Camelyon17 and CivilComments (CC0), "
    "FMoW (custom licence, https://github.com/fMoW/dataset/blob/master/LICENSE)."
)


def discover_tasks(data_dir):
    tasks = {}
    for path in sorted(data_dir.glob("*.pt")):
        is_labels_file = path.stem.endswith("_labels")
        task = path.stem[:-len("_labels")] if is_labels_file else path.stem
        if task not in KNOWN_TASKS:
            raise ValueError(f"Unknown task file: {path}")
        if is_labels_file:
            continue
        labels = path.with_name(f"{task}_labels.pt")
        if not labels.is_file():
            raise ValueError(f"Missing labels file for {path.name}: {labels}")
        tasks[task] = (path, labels)
    return tasks


def stage_file(path, staging):
    target = staging / path.name
    try:
        os.link(path, target)  # the full benchmark is several GB; avoid a second copy
    except OSError:
        shutil.copyfile(path, target)


def load_tensor(path):
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Could not load {path}: {exc}") from exc
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{path}: expected a tensor")
    return tensor


def validate_task(predictions_path, labels_path):
    preds = load_tensor(predictions_path)
    if preds.dtype != torch.float16 or preds.ndim != 3:
        raise ValueError(f"{predictions_path}: expected float16 predictions with shape (H, N, C)")
    h, n, c = preds.shape

    labels = load_tensor(labels_path)
    if labels.dtype != torch.uint8 or labels.shape != (n,):
        raise ValueError(f"{labels_path}: expected uint8 labels with shape ({n},)")
    if labels.numel() and labels.max().item() >= c:
        raise ValueError(f"{labels_path}: label out of range for {c} classes")

    if n:
        count = min(n, 1000)
        indices = torch.arange(count) * n // count
        sums = preds[:, indices, :].sum(dim=-1, dtype=torch.float32)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-2, rtol=0):
            raise ValueError(f"{predictions_path}: sampled softmax rows do not sum to 1")
    return h, n, c


def build_card(repo_id, task_info):
    rows = "\n".join(
        f"| {task} | {h} | {n} | {c} | {size:,} |"
        for task, (h, n, c, size) in task_info.items()
    )
    source_note = ""
    if MODELSELECTOR_TASKS.intersection(task_info):
        source_note = "\nThese ModelSelector prediction files come from the ModelSelector release.\n"
    example_task = next(iter(task_info), "<task>")
    return f"""---
license: cc0-1.0
pretty_name: CODA Active Model Selection Benchmark
tags: [model-selection, active-learning, benchmark]
---

# CODA Active Model Selection Benchmark

Paper: https://arxiv.org/abs/2507.23771

## Files

Each task has two PyTorch files. `<task>.pt` contains float16 softmax predictions
with shape `(H, N, C)` for H models, N points and C classes.
`<task>_labels.pt` contains uint8 labels with shape `(N,)`.

## Included tasks

| Task | H | N | C | File size (bytes, both files) |
| --- | ---: | ---: | ---: | ---: |
{rows}

## Loading

```python
import torch
from huggingface_hub import hf_hub_download

repo_id = {repo_id!r}
task = {example_task!r}
predictions_path = hf_hub_download(repo_id, f"{{task}}.pt", repo_type="dataset")
labels_path = hf_hub_download(repo_id, f"{{task}}_labels.pt", repo_type="dataset")
predictions = torch.load(predictions_path, map_location="cpu", weights_only=True)
labels = torch.load(labels_path, map_location="cpu", weights_only=True)
```

Place both files in the same directory to run the benchmark:

```sh
python main.py --task <task> --data-dir <dir>
```

## Licence

{LICENCE}
{source_note}
## BibTeX

```bibtex
@misc{{kay2025consensusdriven,
  title = {{Consensus-Driven Active Model Selection}},
  author = {{Kay, Justin and Van Horn, Grant and Maji, Subhransu and Sheldon, Daniel and Beery, Sara}},
  year = {{2025}},
  eprint = {{2507.23771}},
  archivePrefix = {{arXiv}},
  url = {{https://arxiv.org/abs/2507.23771}}
}}
```
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, metavar="USER/NAME")
    parser.add_argument("--include-modelselector", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.data_dir.is_dir():
        parser.error(f"Data directory does not exist: {args.data_dir}")

    try:
        tasks = discover_tasks(args.data_dir)
        missing = KNOWN_TASKS.difference(tasks)
        if missing:
            print(f"Warning: missing known tasks: {', '.join(sorted(missing))}", file=sys.stderr)
        skipped = sorted(MODELSELECTOR_TASKS.intersection(tasks)) if not args.include_modelselector else []
        print(f"Skipped tasks: {', '.join(skipped) or 'none'}")
        for task in skipped:
            del tasks[task]

        task_info = {}
        for task, paths in tasks.items():
            dimensions = validate_task(*paths)
            task_info[task] = (*dimensions, sum(path.stat().st_size for path in paths))

        if args.dry_run:
            total_size = sum(info[-1] for info in task_info.values())
            print(f"Dry run for {args.repo_id}")
            print(f"Included tasks: {', '.join(tasks) or 'none'}")
            print(f"Total size: {total_size:,} bytes (tensor files)")
            return

        with tempfile.TemporaryDirectory(prefix="coda-hf-") as staging_dir:
            staging = Path(staging_dir)
            for paths in tasks.values():
                for path in paths:
                    stage_file(path, staging)
            (staging / "README.md").write_text(build_card(args.repo_id, task_info), encoding="utf-8")

            api = huggingface_hub.HfApi()
            api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
            api.upload_folder(
                folder_path=staging,
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message="Add CODA benchmark predictions",
            )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
