import huggingface_hub
import pytest
import torch

from scripts import release_hf_dataset as release


REPO_ID = "owner/coda"


def save_task(data_dir, task, preds=None, labels=None):
    if preds is None:
        preds = torch.full((3, 20, 4), 0.25, dtype=torch.float16)
    if labels is None:
        labels = torch.arange(20, dtype=torch.uint8) % 4
    paths = data_dir / f"{task}.pt", data_dir / f"{task}_labels.pt"
    torch.save(preds, paths[0])
    torch.save(labels, paths[1])
    return paths


def run_release(data_dir, *flags):
    return release.main(["--data-dir", str(data_dir), "--repo-id", REPO_ID, *flags])


def test_discovery(tmp_path):
    wilds = save_task(tmp_path, "iwildcam")
    domainnet = save_task(tmp_path, "sketch_painting")
    (tmp_path / "notes.txt").touch()

    assert release.discover_tasks(tmp_path) == {
        "iwildcam": wilds,
        "sketch_painting": domainnet,
    }


@pytest.mark.parametrize("flag", [(), ("--include-modelselector",)])
def test_dry_run(tmp_path, monkeypatch, capsys, flag):
    wilds = save_task(tmp_path, "iwildcam")
    modelselector = save_task(tmp_path, "pacs")

    def unexpected_api():
        pytest.fail("Dry run constructed HfApi")

    monkeypatch.setattr(huggingface_hub, "HfApi", unexpected_api)
    run_release(tmp_path, "--dry-run", *flag)

    output = capsys.readouterr()
    included = "iwildcam, pacs" if flag else "iwildcam"
    paths = wilds + modelselector if flag else wilds
    expected_size = sum(path.stat().st_size for path in paths)
    assert f"Dry run for {REPO_ID}\n" in output.out
    assert f"Included tasks: {included}\n" in output.out
    assert f"Skipped tasks: {'none' if flag else 'pacs'}\n" in output.out
    assert f"Total size: {expected_size:,} bytes" in output.out
    assert "Warning: missing known tasks:" in output.err
    assert set(tmp_path.iterdir()) == set(wilds + modelselector)


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("preds_dtype", "iwildcam.pt: expected float16"),
        ("labels_dtype", "iwildcam_labels.pt: expected uint8"),
        ("labels_shape", "iwildcam_labels.pt: expected uint8 labels with shape (20,)"),
        ("range", "iwildcam_labels.pt: label out of range"),
        ("missing", "iwildcam_labels.pt"),
        ("unknown", "unknown.pt"),
        ("unknown_labels", "unknown_labels.pt"),
        ("softmax", "iwildcam.pt: sampled softmax rows do not sum to 1"),
    ],
)
def test_invalid_task(tmp_path, capsys, problem, message):
    preds, labels = save_task(tmp_path, "iwildcam")
    if problem == "preds_dtype":
        torch.save(torch.full((3, 20, 4), 0.25), preds)
    elif problem == "labels_dtype":
        torch.save(torch.zeros(20, dtype=torch.long), labels)
    elif problem == "labels_shape":
        torch.save(torch.zeros(19, dtype=torch.uint8), labels)
    elif problem == "range":
        torch.save(torch.full((20,), 4, dtype=torch.uint8), labels)
    elif problem == "missing":
        labels.unlink()
    elif problem == "unknown":
        (tmp_path / "unknown.pt").touch()
    elif problem == "unknown_labels":
        (tmp_path / "unknown_labels.pt").touch()
    elif problem == "softmax":
        torch.save(torch.zeros((3, 20, 4), dtype=torch.float16), preds)

    with pytest.raises(SystemExit) as error:
        run_release(tmp_path, "--dry-run")
    assert error.value.code != 0
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("flags", [(), ("--private", "--include-modelselector")])
def test_upload_and_card(tmp_path, monkeypatch, flags):
    wilds = save_task(tmp_path, "iwildcam")
    modelselector = save_task(tmp_path, "pacs")
    calls = []

    class FakeApi:
        def create_repo(self, *args, **kwargs):
            calls.append(("create", args, kwargs))

        def upload_folder(self, **kwargs):
            folder = kwargs.pop("folder_path")
            calls.append(("upload", folder, kwargs))
            expected_files = wilds + modelselector if flags else wilds
            assert {path.name for path in folder.iterdir()} == {
                "README.md", *(path.name for path in expected_files)
            }
            for path in expected_files:
                assert (folder / path.name).read_bytes() == path.read_bytes()

            card = (folder / "README.md").read_text(encoding="utf-8")
            front_matter = card.split("---", 2)[1]
            assert "license: cc0-1.0" in front_matter
            assert "pretty_name: CODA Active Model Selection Benchmark" in front_matter
            assert "tags: [model-selection, active-learning, benchmark]" in front_matter
            assert "arxiv" not in front_matter
            assert "https://arxiv.org/abs/2507.23771" in card
            assert "| Task | H | N | C |" in card
            size = sum(path.stat().st_size for path in wilds)
            assert f"| iwildcam | 3 | 20 | 4 | {size:,} |" in card
            assert ("| pacs | 3 | 20 | 4 |" in card) == bool(flags)
            assert ("come from the ModelSelector release" in card) == bool(flags)
            assert release.LICENCE in card
            assert "@misc{kay2025consensusdriven," in card
            assert "TODO" not in card
            assert 'map_location="cpu", weights_only=True' in card
            assert "python main.py --task <task> --data-dir <dir>" in card

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    run_release(tmp_path, *flags)

    create, upload = calls
    assert create == (
        "create", (REPO_ID,), {"repo_type": "dataset", "private": bool(flags), "exist_ok": True}
    )
    assert upload[0] == "upload"
    assert upload[1] != tmp_path and not upload[1].exists()
    assert upload[2] == {
        "repo_id": REPO_ID,
        "repo_type": "dataset",
        "commit_message": "Add CODA benchmark predictions",
    }
    assert set(tmp_path.iterdir()) == set(wilds + modelselector)
