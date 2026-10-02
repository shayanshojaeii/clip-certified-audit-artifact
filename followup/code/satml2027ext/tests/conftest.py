"""Shared fixtures: synthetic data roots and the two item-manifest builds.

No test reads the real CIFAR-100 test batch or an Imagenette folder; those are
synthesized under pytest's tmp directory.  The real Phase-3 assignments file,
the Phase-4 consumed list and the EXP-019 item lists are read (never written)
when present, and the affected tests are skipped otherwise.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "satml2027ext"
for entry in (str(ROOT), str(PACKAGE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import datasets_ext  # noqa: E402
import draw_items_ext  # noqa: E402

PHASE3 = ROOT / draw_items_ext.DEFAULT_PHASE3
PHASE4 = ROOT / draw_items_ext.DEFAULT_PHASE4
EXP019 = ROOT / draw_items_ext.DEFAULT_EXP019_ITEMS
CREATED_AT = "2026-09-25T00:00:00+00:00"


def require_real_inputs() -> None:
    for path in (PHASE3, PHASE4, EXP019 / "item_manifest.json"):
        if not path.is_file():
            pytest.skip(f"real input not present locally: {path}")


def make_synthetic_cifar100_test(root: Path, count: int = 4000) -> list[int]:
    """Write cifar-100-python/{test,meta} pickles in torchvision's layout."""

    folder = root / datasets_ext.CIFAR100_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    labels = [index % 100 for index in range(count)]
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=(count, 3 * 32 * 32), dtype=np.uint8)
    with (folder / datasets_ext.CIFAR100_TEST_FILE).open("wb") as handle:
        pickle.dump({b"fine_labels": labels, b"coarse_labels": [0] * count, b"data": data,
                     b"filenames": [f"synthetic_{index}.png".encode() for index in range(count)]}, handle)
    with (folder / datasets_ext.CIFAR100_META_FILE).open("wb") as handle:
        pickle.dump({b"fine_label_names": [f"class_{index:03d}".encode() for index in range(100)]}, handle)
    return labels


def make_synthetic_imagenette(root: Path, per_class: dict[str, int]) -> None:
    """Write imagenette2-320/{split}/{wnid}/*.JPEG (1x1 images) for the ten wnids."""

    from PIL import Image

    for split, count in per_class.items():
        for wnid in datasets_ext.IMAGENETTE_WNIDS:
            folder = root / datasets_ext.IMAGENETTE_FOLDER / split / wnid
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                Image.new("RGB", (1, 1), (index % 256, 0, 0)).save(folder / f"{wnid}_{index}.JPEG", format="JPEG")


def draw_args(out: Path, data_root: Path, *, skip_unavailable: bool, allow_synthetic: bool) -> argparse.Namespace:
    return argparse.Namespace(
        phase3_assignments=PHASE3,
        phase4_assignments=PHASE4,
        exp019_items=EXP019,
        data_root=data_root,
        out=out,
        skip_unavailable=skip_unavailable,
        created_at=CREATED_AT,
        allow_synthetic=allow_synthetic,
    )


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synthetic_data")
    make_synthetic_cifar100_test(root)
    make_synthetic_imagenette(root, {"val": 310, "train": 55})
    return root


@pytest.fixture(scope="session")
def empty_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("empty_data")


@pytest.fixture(scope="session")
def full_build(tmp_path_factory, synthetic_root):
    """All lists written (synthetic CIFAR-100 test and Imagenette)."""

    require_real_inputs()
    out = tmp_path_factory.mktemp("items_full")
    manifest = draw_items_ext.build(draw_args(out, synthetic_root, skip_unavailable=False, allow_synthetic=True))
    return out, manifest


@pytest.fixture(scope="session")
def real_only_build(tmp_path_factory, empty_root):
    """Only the lists that need the assignments file; the rest pending."""

    require_real_inputs()
    out = tmp_path_factory.mktemp("items_real_only")
    manifest = draw_items_ext.build(draw_args(out, empty_root, skip_unavailable=True, allow_synthetic=False))
    return out, manifest
