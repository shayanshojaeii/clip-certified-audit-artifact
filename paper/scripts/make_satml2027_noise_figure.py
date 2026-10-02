#!/usr/bin/env python3
"""Appendix figure (D-148): one registered development image per EXP-021B dataset at several noise levels.

Choice of images (a fixed rule, not a selection by outcome): the first EXP-021B development item, in registered order,
of a fixed class per dataset (results/satml2027ext/items/exp021b__<dataset>__development.csv): class 0 for CIFAR-100
(apple) and EuroSAT (annual crop); class 8 (golf ball) for Imagenette, because the first class-0 image shows an
identifiable person, and the paper shows no identifiable person. These items were decoded during the registered bank
preparation, so the figure reads no unaccessed reserve, evaluation, official-test or sealed item.

Coordinates: OpenCLIP's evaluation transform without its final Normalize step (bicubic resize of the shorter side to
224, centre crop, tensor in [0, 1]); this is the space in which the adversary and the Gaussian noise act (Section II).
Noise: one fixed display draw z ~ N(0, I) (seed below), shown as x + sigma z for each sigma. The display clips to
[0, 1]; the certified classifier sees unclipped samples. Nothing here touches a model or a result file.

Writes paper/satml2027/generated/fig_noise_<dataset>_<k>.png (224 x 224 each, unscaled) and fig_noise_sources.json.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "results/satml2027ext/items"
OUT = ROOT / "paper/satml2027/generated"
SIGMAS = (0.0, 0.12, 0.25, 0.5)
DISPLAY_SEED = 20260928
CIFAR_ARCHIVE = ROOT / "data/cifar-100-python.tar.gz"
DATASETS = (  # (cell dataset id of the item list, display name, fixed class, local image root for file-backed items)
    ("cifar100_test", "cifar100", 0, None),          # development items: CIFAR-100 training images (reserve)
    ("eurosat_sealed", "eurosat", 0, ROOT / "data"),  # development items: the EuroSAT reserve, never the sealed holdout
    ("imagenette", "imagenette", 8, ROOT / "data_ext"),  # golf ball: the first class-0 image shows a person
)


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def first_development_item(dataset: str, label: int) -> dict[str, str]:
    with (ITEMS / f"exp021b__{dataset}__development.csv").open(encoding="utf-8", newline="") as handle:
        return next(row for row in csv.DictReader(handle) if int(row["label"]) == label)


def load_image(dataset: str, row: dict[str, str], root: Path | None) -> tuple[Image.Image, str, str]:
    """(RGB image, source description, sha256 of the bytes it came from)."""

    if dataset == "cifar100_test":  # development items are CIFAR-100 training images (reserved split)
        index = int(row["dataset_index"])
        if not row["item_id"].startswith("cifar100-train-") or int(row["item_id"].rsplit("-", 1)[1]) != index:
            raise RuntimeError(f"unexpected CIFAR-100 development item {row}")
        with tarfile.open(CIFAR_ARCHIVE, "r:gz") as archive:
            blob = archive.extractfile("cifar-100-python/train").read()  # streamed in memory, not extracted
        batch = pickle.loads(blob, encoding="bytes")
        if int(batch[b"fine_labels"][index]) != int(row["label"]):
            raise RuntimeError("CIFAR-100 label differs from the registered development list")
        pixels = batch[b"data"][index].reshape(3, 32, 32).transpose(1, 2, 0)
        return Image.fromarray(pixels, "RGB"), f"{CIFAR_ARCHIVE.name}:cifar-100-python/train[{index}]", sha256_bytes(blob)
    path = root / row["item_id"]
    blob = path.read_bytes()
    return Image.open(io.BytesIO(blob)).convert("RGB"), row["item_id"], sha256_bytes(blob)


def main() -> int:
    import open_clip  # the transform only; no model is loaded

    transform = open_clip.image_transform(224, is_train=False)
    to_pixels = [step for step in transform.transforms if type(step).__name__ != "Normalize"]
    if len(to_pixels) != len(transform.transforms) - 1:
        raise RuntimeError("expected exactly one Normalize step in the OpenCLIP evaluation transform")
    rng = np.random.default_rng(DISPLAY_SEED)
    z = rng.standard_normal((3, 224, 224))
    OUT.mkdir(parents=True, exist_ok=True)
    record = {"generator": "scripts/make_satml2027_noise_figure.py", "open_clip": open_clip.__version__,
              "transform": [str(step) for step in to_pixels], "display_seed": DISPLAY_SEED, "sigmas": list(SIGMAS),
              "rule": ("first EXP-021B development item in registered order of a fixed class per dataset: class 0 for CIFAR-100 and EuroSAT, class 8 (golf ball) for Imagenette because the first class-0 image shows an identifiable person"), "images": []}
    for dataset, display, label, root in DATASETS:
        row = first_development_item(dataset, label)
        if "sealed" in row["source_split"] or "test" in row["source_split"]:
            raise RuntimeError(f"{dataset}: the first development item comes from {row['source_split']}")
        image, source, digest = load_image(dataset, row, root)
        x = image
        for step in to_pixels:
            x = step(x)
        x = x.numpy().astype(np.float64)
        if x.shape != (3, 224, 224) or x.min() < 0.0 or x.max() > 1.0:
            raise RuntimeError(f"{dataset}: unexpected pixel tensor {x.shape} [{x.min()}, {x.max()}]")
        names = []
        for k, sigma in enumerate(SIGMAS):
            shown = np.clip(x + sigma * z, 0.0, 1.0)  # display only; the classifier sees unclipped samples
            tile = Image.fromarray(np.rint(shown.transpose(1, 2, 0) * 255.0).astype(np.uint8), "RGB")
            name = f"fig_noise_{display}_{k}.png"
            tile.save(OUT / name, optimize=True)
            names.append(name)
        record["images"].append({"dataset": display, "cell_dataset_id": dataset, "item_id": row["item_id"], "label": int(row["label"]),
                                 "source_split": row["source_split"], "source": source, "source_sha256": digest,
                                 "tiles": names})
        print(f"{dataset}: {row['item_id']} -> {len(names)} tiles")
    (OUT / "fig_noise_sources.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
