"""Compare an exact reviewer Gate-N2 reconstruction with the primary output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reconstruction", type=Path)
    parser.add_argument(
        "--primary",
        type=Path,
        default=Path("artifacts/negative_paper/gate_n2_simultaneous_analysis_v1.json"),
    )
    args = parser.parse_args()
    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    reconstructed = json.loads(args.reconstruction.read_text(encoding="utf-8"))
    if reconstructed != primary:
        raise RuntimeError("reviewer reconstruction differs from the primary Gate-N2 report")
    print("PASS reviewer reconstruction is structurally and numerically identical")
    print(f"primary_sha256={sha256_file(args.primary)}")
    print(f"reconstruction_sha256={sha256_file(args.reconstruction)}")


if __name__ == "__main__":
    main()
