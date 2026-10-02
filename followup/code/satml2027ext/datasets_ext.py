"""Dataset access for the EXP-021 follow-up study, with frozen item identities.

Three dataset ids are added on top of ``satml2027/common/datasets.py``:

  cifar100_test   -> "cifar100-test-{index}"
                     torchvision ``CIFAR100(train=False)`` layout, i.e. the pickled
                     ``cifar-100-python/test`` batch (10,000 items).  The batch is
                     read directly (index order is torchvision's, because torchvision
                     reads the same single file in the same order), so torch is never
                     required.  By default the file's MD5 must equal the value that
                     torchvision checks (``require_official``).
  eurosat_sealed  -> "eurosat/2750/{Class}/{file}"
                     the 1,000 Phase-3 ``final_test_sealed`` rows of
                     ``artifacts/day14/phase3_splits_v1/assignments.csv`` over the
                     existing EuroSAT folder.  Listing the ids needs no image access.
  imagenette      -> "imagenette2-320/{split}/{wnid}/{file}"
                     the fastai imagenette2-320 layout ``train/<wnid>/*.JPEG`` and
                     ``val/<wnid>/*.JPEG``; labels follow the canonical wnid order
                     below (which coincides with sorted wnid order).

``load_dataset_ext(dataset_id, root, transform)`` returns ``(dataset, item_ids,
labels)`` in the convention of the existing loader.  Nothing here downloads.
The official CIFAR-100 test split and the sealed EuroSAT holdout are consumed
by EXP-021B and must not be used for selection afterwards (plan Section 5).
"""

from __future__ import annotations

import csv
import hashlib
import pickle
from pathlib import Path
from typing import Any, Sequence

# --------------------------------------------------------------------------- #
# Frozen constants
# --------------------------------------------------------------------------- #

IMAGENETTE_FOLDER = "imagenette2-320"
IMAGENETTE_CLASSES: tuple[tuple[str, str], ...] = (
    ("n01440764", "tench"),
    ("n02102040", "English springer"),
    ("n02979186", "cassette player"),
    ("n03000684", "chain saw"),
    ("n03028079", "church"),
    ("n03394916", "French horn"),
    ("n03417042", "garbage truck"),
    ("n03425413", "gas pump"),
    ("n03445777", "golf ball"),
    ("n03888257", "parachute"),
)
IMAGENETTE_WNIDS: tuple[str, ...] = tuple(wnid for wnid, _ in IMAGENETTE_CLASSES)
IMAGENETTE_NAME_BY_WNID: dict[str, str] = dict(IMAGENETTE_CLASSES)
IMAGENETTE_SPLITS = ("train", "val")
# Official image counts of imagenette2-320 (identical across the three sizes).
IMAGENETTE_OFFICIAL_COUNTS = {"train": 9469, "val": 3925}
IMAGENETTE_DOWNLOAD_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
IMAGENETTE_DOWNLOAD_SIZE_NOTE = "about 341 MB compressed (fastai README figure); unpacks to imagenette2-320/{train,val}/<wnid>/*.JPEG"
IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png")

CIFAR100_FOLDER = "cifar-100-python"
CIFAR100_TEST_FILE = "test"
CIFAR100_META_FILE = "meta"
# The MD5 values torchvision.datasets.CIFAR100 verifies (test_list / meta).
CIFAR100_TEST_MD5 = "f0ef6b0ae62326f3e7ffdfab6717acfc"
CIFAR100_META_MD5 = "7973b15100ade9c7d40fb424638fde48"
CIFAR100_TEST_COUNT = 10000
CIFAR100_CLASS_COUNT = 100

EUROSAT_FOLDER = "eurosat/2750"
EUROSAT_CLASS_FOLDERS: tuple[str, ...] = (
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
)
EUROSAT_SEALED_ROLE = "final_test_sealed"
EUROSAT_SEALED_COUNT = 1000

DATASET_REGISTRY_EXT: dict[str, dict[str, Any]] = {
    "cifar100_test": {
        "class_count": CIFAR100_CLASS_COUNT,
        "source_dataset_id": "cifar100",
        "split": "official test split (torchvision CIFAR100 train=False)",
        "item_id_pattern": "cifar100-test-{index}",
        "consumed_once": True,
    },
    "eurosat_sealed": {
        "class_count": 10,
        "source_dataset_id": "eurosat",
        "split": "Phase-3 final_test_sealed (1,000 ids, 100 per class)",
        "item_id_pattern": "eurosat/2750/{Class}/{file}",
        "consumed_once": True,
    },
    "imagenette": {
        "class_count": 10,
        "source_dataset_id": "imagenette",
        "split": "imagenette2-320 val (evaluation) and train (development, control-train)",
        "item_id_pattern": "imagenette2-320/{split}/{wnid}/{file}",
        "consumed_once": False,
    },
}


class DatasetUnavailableError(FileNotFoundError):
    """Raised when pixels or labels are not present locally (fail closed)."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_checked(path: Path, what: str) -> Path:
    try:
        if not path.is_file():
            raise DatasetUnavailableError(f"{what} is not present locally: {path}")
        with path.open("rb"):
            pass
    except PermissionError as error:
        raise DatasetUnavailableError(f"{what} exists but is not readable (permission denied): {path}") from error
    return path


def _load_pickle(path: Path) -> dict[bytes, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="bytes")


# --------------------------------------------------------------------------- #
# CIFAR-100 official test split
# --------------------------------------------------------------------------- #


def cifar100_test_batch_path(root: str | Path) -> Path:
    return Path(root) / CIFAR100_FOLDER / CIFAR100_TEST_FILE


def cifar100_test_labels(root: str | Path, *, require_official: bool = True) -> list[int]:
    """Fine labels of the official test split in torchvision index order.

    Fails closed (``DatasetUnavailableError``) when the pickled batch is absent
    or unreadable, and when ``require_official`` is set and its MD5 differs from
    the value torchvision verifies.  ``require_official=False`` is for synthetic
    test fixtures only and must never be used to produce registered lists.
    """

    path = _open_checked(cifar100_test_batch_path(root), "CIFAR-100 official test batch")
    if require_official:
        digest = _md5(path)
        if digest != CIFAR100_TEST_MD5:
            raise DatasetUnavailableError(
                f"CIFAR-100 test batch MD5 {digest} differs from torchvision's {CIFAR100_TEST_MD5}: {path}"
            )
    batch = _load_pickle(path)
    if b"fine_labels" not in batch:
        raise DatasetUnavailableError(f"CIFAR-100 test batch lacks fine_labels: {path}")
    labels = [int(value) for value in batch[b"fine_labels"]]
    if require_official and len(labels) != CIFAR100_TEST_COUNT:
        raise DatasetUnavailableError(f"CIFAR-100 test batch has {len(labels)} labels, expected {CIFAR100_TEST_COUNT}")
    if any(not 0 <= value < CIFAR100_CLASS_COUNT for value in labels):
        raise DatasetUnavailableError("CIFAR-100 test labels outside [0, 100)")
    return labels


def cifar100_test_class_names(root: str | Path, *, require_official: bool = True) -> list[str]:
    path = _open_checked(Path(root) / CIFAR100_FOLDER / CIFAR100_META_FILE, "CIFAR-100 meta file")
    if require_official and _md5(path) != CIFAR100_META_MD5:
        raise DatasetUnavailableError(f"CIFAR-100 meta MD5 differs from torchvision's {CIFAR100_META_MD5}: {path}")
    meta = _load_pickle(path)
    names = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in meta[b"fine_label_names"]]
    if len(names) != CIFAR100_CLASS_COUNT:
        raise DatasetUnavailableError(f"CIFAR-100 meta lists {len(names)} fine classes, expected {CIFAR100_CLASS_COUNT}")
    return names


class Cifar100TestDataset:
    """The official CIFAR-100 test split, indexed exactly as torchvision does.

    ``dataset[i]`` returns ``(transform(PIL image), label)``.  ``classes`` and
    ``targets`` mirror the torchvision attributes used by the existing code.
    """

    def __init__(self, root: str | Path, transform=None, *, require_official: bool = True) -> None:
        import numpy as np

        self.root = Path(root)
        self.transform = transform
        self.targets = cifar100_test_labels(self.root, require_official=require_official)
        self.classes = cifar100_test_class_names(self.root, require_official=require_official)
        batch = _load_pickle(cifar100_test_batch_path(self.root))
        data = np.asarray(batch[b"data"], dtype=np.uint8)
        if data.shape != (len(self.targets), 3 * 32 * 32):
            raise DatasetUnavailableError(f"CIFAR-100 test data has shape {data.shape}")
        self.data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # HWC, torchvision layout

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        from PIL import Image

        image = Image.fromarray(self.data[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


# --------------------------------------------------------------------------- #
# Folder datasets (EuroSAT, Imagenette)
# --------------------------------------------------------------------------- #


def list_folder_images(split_dir: Path, class_folders: Sequence[str], *, strict_folders: bool) -> list[tuple[str, str, int]]:
    """Return ``[(class_folder, file_name, label)]`` in canonical class order, sorted names."""

    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise DatasetUnavailableError(f"image folder is not present locally: {split_dir}")
    present = sorted(entry.name for entry in split_dir.iterdir() if entry.is_dir())
    missing = [name for name in class_folders if name not in present]
    if missing:
        raise DatasetUnavailableError(f"{split_dir} lacks class folders {missing}")
    extra = [name for name in present if name not in class_folders]
    if strict_folders and extra:
        raise DatasetUnavailableError(f"{split_dir} has unexpected class folders {extra}")
    records: list[tuple[str, str, int]] = []
    for label, folder in enumerate(class_folders):
        names = sorted(
            entry.name for entry in (split_dir / folder).iterdir()
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not names:
            raise DatasetUnavailableError(f"{split_dir / folder} contains no images")
        records.extend((folder, name, label) for name in names)
    return records


class ImageFolderDataset:
    """Minimal PIL-backed folder dataset with an explicit class order.

    ``samples`` is ``[(absolute path, label)]``; ``classes`` is the class-folder
    order that defines the labels.  Torch is not required.
    """

    def __init__(self, samples: Sequence[tuple[Path, int]], classes: Sequence[str], transform=None) -> None:
        self.samples = [(Path(path), int(label)) for path, label in samples]
        self.classes = list(classes)
        self.transform = transform
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        path, label = self.samples[index]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def imagenette_split_items(root: str | Path, split: str, *, require_official: bool = True) -> list[dict[str, str]]:
    """Rows ``{dataset_id, item_id, dataset_index, label}`` of one imagenette2-320 split.

    Fails closed when the folder is absent.  With ``require_official`` the split
    must contain exactly the ten wnids and the official image count.
    """

    if split not in IMAGENETTE_SPLITS:
        raise ValueError(f"imagenette split must be one of {IMAGENETTE_SPLITS}, got {split!r}")
    split_dir = Path(root) / IMAGENETTE_FOLDER / split
    records = list_folder_images(split_dir, IMAGENETTE_WNIDS, strict_folders=require_official)
    if require_official and len(records) != IMAGENETTE_OFFICIAL_COUNTS[split]:
        raise DatasetUnavailableError(
            f"{split_dir} holds {len(records)} images; the official imagenette2-320 {split} split holds "
            f"{IMAGENETTE_OFFICIAL_COUNTS[split]}"
        )
    return [
        {
            "dataset_id": "imagenette",
            "item_id": f"{IMAGENETTE_FOLDER}/{split}/{wnid}/{name}",
            "dataset_index": str(index),
            "label": str(label),
        }
        for index, (wnid, name, label) in enumerate(records)
    ]


def eurosat_sealed_items(phase3_assignments: str | Path) -> list[dict[str, str]]:
    """The 1,000 sealed EuroSAT rows in assignments-file order (ids only, no pixels)."""

    path = Path(phase3_assignments)
    if not path.is_file():
        raise DatasetUnavailableError(f"Phase-3 assignments file is not present: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["dataset_id"] == "eurosat" and row["phase3_role"] == EUROSAT_SEALED_ROLE]
    if len(rows) != EUROSAT_SEALED_COUNT:
        raise RuntimeError(f"expected {EUROSAT_SEALED_COUNT} sealed EuroSAT rows, found {len(rows)}")
    per_class: dict[str, int] = {}
    for row in rows:
        item_id = row["item_id"]
        parts = item_id.split("/")
        if len(parts) != 4 or "/".join(parts[:2]) != EUROSAT_FOLDER or parts[2] not in EUROSAT_CLASS_FOLDERS:
            raise RuntimeError(f"unexpected sealed EuroSAT item id {item_id}")
        if int(row["label"]) != EUROSAT_CLASS_FOLDERS.index(parts[2]):
            raise RuntimeError(f"sealed EuroSAT label disagrees with its class folder: {item_id}")
        per_class[row["label"]] = per_class.get(row["label"], 0) + 1
    if sorted(per_class.values()) != [100] * 10:
        raise RuntimeError(f"sealed EuroSAT rows are not 100 per class: {per_class}")
    return [{"dataset_id": "eurosat", "item_id": row["item_id"], "dataset_index": row["dataset_index"], "label": row["label"]}
            for row in rows]


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def load_dataset_ext(
    dataset_id: str,
    root: str | Path,
    transform=None,
    *,
    phase3_assignments: str | Path | None = None,
    split: str | None = None,
    require_official: bool = True,
):
    """Return ``(dataset, item_ids, labels)`` with download disabled.

    ``eurosat_sealed`` needs ``phase3_assignments``; ``imagenette`` needs
    ``split`` in {"train", "val"}.  ``require_official=False`` disables the
    integrity checks and is reserved for synthetic test fixtures.
    """

    base = Path(root)
    if dataset_id == "cifar100_test":
        dataset = Cifar100TestDataset(base, transform, require_official=require_official)
        item_ids = [f"cifar100-test-{index}" for index in range(len(dataset))]
        labels = list(dataset.targets)
    elif dataset_id == "eurosat_sealed":
        if phase3_assignments is None:
            raise ValueError("eurosat_sealed requires phase3_assignments")
        rows = eurosat_sealed_items(phase3_assignments)
        samples = []
        missing = []
        for row in rows:
            path = base / row["item_id"]
            if not path.is_file():
                missing.append(row["item_id"])
            samples.append((path, int(row["label"])))
        if missing:
            raise DatasetUnavailableError(
                f"{len(missing)} of {len(rows)} sealed EuroSAT images are absent under {base} (first: {missing[0]})"
            )
        dataset = ImageFolderDataset(samples, EUROSAT_CLASS_FOLDERS, transform)
        item_ids = [row["item_id"] for row in rows]
        labels = [int(row["label"]) for row in rows]
    elif dataset_id == "imagenette":
        if split is None:
            raise ValueError("imagenette requires split='train' or 'val'")
        rows = imagenette_split_items(base, split, require_official=require_official)
        samples = [(base / row["item_id"], int(row["label"])) for row in rows]
        dataset = ImageFolderDataset(samples, IMAGENETTE_WNIDS, transform)
        item_ids = [row["item_id"] for row in rows]
        labels = [int(row["label"]) for row in rows]
    else:
        raise ValueError(f"unsupported dataset {dataset_id}; known: {sorted(DATASET_REGISTRY_EXT)}")
    if len(item_ids) != len(labels) or len(item_ids) != len(dataset):
        raise RuntimeError("item ids, labels and dataset length disagree")
    return dataset, item_ids, labels


def class_names_ext(dataset, prompt_config: dict[str, Any]) -> list[str]:
    """Prompt class names in dataset class order; supports the wnid mapping."""

    mapping = prompt_config.get("class_name_by_wnid")
    if mapping is not None:
        if set(mapping) != set(dataset.classes):
            raise RuntimeError("prompt wnid mapping does not match dataset classes")
        return [mapping[wnid] for wnid in dataset.classes]
    import sys

    satml_dir = Path(__file__).resolve().parents[1] / "satml2027"
    if str(satml_dir) not in sys.path:
        sys.path.insert(0, str(satml_dir))
    from common.datasets import class_names  # frozen helper; torch-free at import

    return class_names(dataset, prompt_config)
