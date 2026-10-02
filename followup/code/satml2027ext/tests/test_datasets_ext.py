from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from conftest import EXP019, PHASE3, ROOT, make_synthetic_cifar100_test, make_synthetic_imagenette  # noqa: F401

import datasets_ext as dx


def test_module_never_imports_torch():
    source = (ROOT / "satml2027ext" / "datasets_ext.py").read_text(encoding="utf-8")
    statements = [line.split("#")[0].strip() for line in source.splitlines() if line.strip().startswith(("import ", "from "))]
    assert not [line for line in statements if "torch" in line], statements
    # The sys.modules check must run in a fresh interpreter: other test modules import torch.
    import subprocess
    code = "import sys; sys.path.insert(0, %r); import satml2027ext.datasets_ext; print('torch' in sys.modules)" % str(ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout


def test_registry_and_constants():
    assert set(dx.DATASET_REGISTRY_EXT) == {"cifar100_test", "eurosat_sealed", "imagenette"}
    assert len(dx.IMAGENETTE_WNIDS) == 10 and list(dx.IMAGENETTE_WNIDS) == sorted(dx.IMAGENETTE_WNIDS)
    assert dx.IMAGENETTE_NAME_BY_WNID["n01440764"] == "tench"
    assert dx.IMAGENETTE_NAME_BY_WNID["n03888257"] == "parachute"
    assert dx.IMAGENETTE_OFFICIAL_COUNTS == {"train": 9469, "val": 3925}
    assert dx.CIFAR100_TEST_MD5 == "f0ef6b0ae62326f3e7ffdfab6717acfc"


# ---------------------------------------------------------------- CIFAR-100 test


def test_cifar100_test_labels_synthetic(tmp_path):
    labels = make_synthetic_cifar100_test(tmp_path, count=300)
    assert dx.cifar100_test_labels(tmp_path, require_official=False) == labels
    with pytest.raises(dx.DatasetUnavailableError, match="MD5"):
        dx.cifar100_test_labels(tmp_path, require_official=True)


def test_cifar100_test_fails_closed_when_absent(tmp_path):
    with pytest.raises(dx.DatasetUnavailableError, match="not present"):
        dx.cifar100_test_labels(tmp_path)
    with pytest.raises(dx.DatasetUnavailableError):
        dx.load_dataset_ext("cifar100_test", tmp_path)


def test_load_cifar100_test_dataset_synthetic(tmp_path):
    labels = make_synthetic_cifar100_test(tmp_path, count=120)
    dataset, item_ids, got = dx.load_dataset_ext("cifar100_test", tmp_path, require_official=False)
    assert item_ids == [f"cifar100-test-{index}" for index in range(120)]
    assert got == labels and dataset.targets == labels and len(dataset) == 120
    assert len(dataset.classes) == 100 and dataset.classes[7] == "class_007"
    image, label = dataset[5]
    assert image.size == (32, 32) and image.mode == "RGB" and label == labels[5]
    dataset_t, _, _ = dx.load_dataset_ext("cifar100_test", tmp_path, transform=lambda image: image.size, require_official=False)
    assert dataset_t[0][0] == (32, 32)


# ---------------------------------------------------------------- Imagenette


def test_imagenette_synthetic_items_and_loader(tmp_path):
    make_synthetic_imagenette(tmp_path, {"val": 3, "train": 2})
    rows = dx.imagenette_split_items(tmp_path, "val", require_official=False)
    assert len(rows) == 30
    assert [row["label"] for row in rows[:3]] == ["0", "0", "0"] and rows[-1]["label"] == "9"
    assert rows[0]["item_id"] == "imagenette2-320/val/n01440764/n01440764_0.JPEG"
    assert [row["dataset_index"] for row in rows] == [str(index) for index in range(30)]
    dataset, item_ids, labels = dx.load_dataset_ext("imagenette", tmp_path, split="train", require_official=False)
    assert len(dataset) == 20 and item_ids[0].startswith("imagenette2-320/train/n01440764/")
    assert labels == [index // 2 for index in range(20)] and dataset.classes == list(dx.IMAGENETTE_WNIDS)
    image, label = dataset[19]
    assert image.size == (1, 1) and label == 9
    with pytest.raises(dx.DatasetUnavailableError, match="official"):
        dx.imagenette_split_items(tmp_path, "val", require_official=True)
    with pytest.raises(ValueError):
        dx.load_dataset_ext("imagenette", tmp_path)


def test_imagenette_fails_closed(tmp_path):
    with pytest.raises(dx.DatasetUnavailableError, match="not present"):
        dx.imagenette_split_items(tmp_path, "val")
    make_synthetic_imagenette(tmp_path, {"val": 1})
    (tmp_path / dx.IMAGENETTE_FOLDER / "val" / "n99999999").mkdir()
    with pytest.raises(dx.DatasetUnavailableError, match="unexpected"):
        dx.imagenette_split_items(tmp_path, "val", require_official=True)
    assert len(dx.imagenette_split_items(tmp_path, "val", require_official=False)) == 10
    (tmp_path / dx.IMAGENETTE_FOLDER / "val" / "n01440764").rename(tmp_path / "gone")
    with pytest.raises(dx.DatasetUnavailableError, match="lacks"):
        dx.imagenette_split_items(tmp_path, "val", require_official=False)


def test_imagenette_prompt_config_maps_canonical_names(tmp_path):
    config = json.loads((ROOT / "configs/satml2027ext/prompts/imagenette_openai_v1.json").read_text(encoding="utf-8"))
    assert len(config["templates"]) == 7 and all("{class_name}" in template for template in config["templates"])
    assert config["class_name_by_wnid"] == dx.IMAGENETTE_NAME_BY_WNID
    make_synthetic_imagenette(tmp_path, {"val": 1})
    dataset, _, _ = dx.load_dataset_ext("imagenette", tmp_path, split="val", require_official=False)
    assert dx.class_names_ext(dataset, config) == [name for _, name in dx.IMAGENETTE_CLASSES]
    from common.prompts import render_prompts

    rendered = render_prompts(dx.class_names_ext(dataset, config), config["templates"])
    assert rendered[1][0] == "a photo of a English springer" and len(rendered) == 10


# ---------------------------------------------------------------- EuroSAT sealed


def test_eurosat_sealed_ids_from_assignments():
    if not PHASE3.is_file():
        pytest.skip("Phase-3 assignments not present")
    rows = dx.eurosat_sealed_items(PHASE3)
    assert len(rows) == 1000
    assert all(row["item_id"].startswith("eurosat/2750/") for row in rows)
    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    assert sorted(counts.values()) == [100] * 10
    assert dx.EUROSAT_CLASS_FOLDERS[int(rows[0]["label"])] == rows[0]["item_id"].split("/")[2]


def test_eurosat_sealed_loader_fails_closed_without_pixels(tmp_path):
    if not PHASE3.is_file():
        pytest.skip("Phase-3 assignments not present")
    with pytest.raises(dx.DatasetUnavailableError, match="absent"):
        dx.load_dataset_ext("eurosat_sealed", tmp_path, phase3_assignments=PHASE3)
    with pytest.raises(ValueError):
        dx.load_dataset_ext("eurosat_sealed", tmp_path)


def test_eurosat_sealed_loader_synthetic_pixels(tmp_path):
    if not PHASE3.is_file():
        pytest.skip("Phase-3 assignments not present")
    from PIL import Image

    rows = dx.eurosat_sealed_items(PHASE3)
    for row in rows:
        path = tmp_path / row["item_id"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1, 1)).save(path, format="JPEG")
    dataset, item_ids, labels = dx.load_dataset_ext("eurosat_sealed", tmp_path, phase3_assignments=PHASE3)
    assert item_ids == [row["item_id"] for row in rows] and labels == [int(row["label"]) for row in rows]
    assert dataset.classes == list(dx.EUROSAT_CLASS_FOLDERS) and len(dataset) == 1000
    eurosat_config = json.loads((ROOT / "configs/prompts/eurosat_openai_ensemble_v1.json").read_text(encoding="utf-8"))
    assert dx.class_names_ext(dataset, eurosat_config)[0] == "annual crop land"


def test_unknown_dataset_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        dx.load_dataset_ext("cifar100", tmp_path)
