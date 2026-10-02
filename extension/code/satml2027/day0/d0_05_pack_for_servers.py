#!/usr/bin/env python3
"""Day 0, step 5 - build the transfer bundle and print the exact copy commands.

Run on the LAPTOP, after d0_00 .. d0_04 have succeeded.  It zips the three
directories both servers need (code, registrations, drawn item lists), reports
the size of every other thing that must reach a server, and prints the scp /
rsync commands with the real resolved paths filled in.

  python satml2027/day0/d0_05_pack_for_servers.py --user you --host-a a.example --host-b b.example
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.hashing import read_json, sha256_file, write_json_atomic  # noqa: E402

CODE_MEMBERS = [
    "satml2027", "configs/satml2027", "configs/prompts",
    "configs/phase3_calibration_protocol_v1.json",
    "configs/phase3_noisy_margin_interface_v1.json",
    "artifacts/day14/phase3_splits_v1/manifest.json",
    "artifacts/day14/phase3_splits_v1/exclusion_manifest.json",
    "certification", "interventions", "results/satml2027/items",
]
EXCLUDE_PARTS = {"__pycache__", ".git", ".ipynb_checkpoints"}
REMOTE_ROOT = "~/certmg"


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _human(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} TiB"


def build_code_zip(root: Path, target: Path) -> tuple[Path, list[str]]:
    target.parent.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in CODE_MEMBERS:
            source = root / member
            if not source.exists():
                continue
            for path in sorted(source.rglob("*")) if source.is_dir() else [source]:
                if not path.is_file() or EXCLUDE_PARTS & set(path.parts):
                    continue
                name = path.relative_to(root).as_posix()
                archive.write(path, name)
                written.append(name)
    return target, written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--paths", type=Path, default=None, help="results/satml2027/paths.json")
    parser.add_argument("--out", type=Path, default=None, help="directory for the transfer bundle")
    parser.add_argument("--user", default="USER")
    parser.add_argument("--host-a", default="SERVER_A", help="the 1x RTX 4090 host")
    parser.add_argument("--host-b", default="SERVER_B", help="the 4x RTX 3090 host")
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    parser.add_argument("--pull-root", type=Path, default=None,
                        help="local directory under which serverA/serverB result pulls are written")
    args = parser.parse_args()

    root = args.root.resolve()
    paths_file = args.paths or (root / "results" / "satml2027" / "paths.json")
    if not paths_file.is_file():
        print(f"missing {paths_file}; run d0_00_locate.py first", file=sys.stderr)
        return 1
    paths = read_json(paths_file)
    out = (args.out or (root / "transfer")).resolve()

    missing = [key for key in ("data_root", "exp016_results", "exp017_results") if not paths.get(key)]
    if missing:
        print("paths.json is incomplete: " + ", ".join(missing), file=sys.stderr)
        return 1

    for member in CODE_MEMBERS:
        if not (root / member).exists():
            print(f"missing {member}; run the earlier Day-0 steps first", file=sys.stderr)
            return 1

    zip_path, members = build_code_zip(root, out / "satml2027_code.zip")
    digest = sha256_file(zip_path)
    print(f"code bundle : {zip_path}")
    print(f"              {len(members)} files, {_human(zip_path.stat().st_size)}, sha256 {digest[:16]}...")

    data_root = root / paths["data_root"]
    exp016 = root / paths["exp016_results"]
    exp017 = root / paths["exp017_results"]
    datasets = {
        "cifar-100-python": data_root / "cifar-100-python",
        "cifar-10-batches-py": data_root / "cifar-10-batches-py",
        "eurosat": data_root / "eurosat",
    }

    print("\nmust also reach BOTH servers:")
    rows: list[tuple[str, Path, int]] = []
    for name, path in datasets.items():
        if path.exists():
            rows.append((f"data/{name}", path, _size(path)))
        else:
            print(f"  [MISSING] data/{name} - not on this laptop yet")
    for label, path in (("EXP-016", exp016), ("EXP-017", exp017)):
        rows.append((f"{label} results", path, _size(path)))
    for name, path, size in rows:
        print(f"  {name:28s} {_human(size):>12s}  {path}")
    print(f"  {'TOTAL':28s} {_human(sum(size for _, _, size in rows)):>12s}")

    write_json_atomic(out / "transfer_manifest.json", {
        "schema_version": "satml2027.transfer.v1",
        "code_zip": zip_path.name,
        "code_zip_sha256": digest,
        "code_file_count": len(members),
        "payloads": [{"name": name, "local_path": str(path), "bytes": size} for name, path, size in rows],
        "remote_root": args.remote_root,
    })

    user, host_a, host_b, remote = args.user, args.host_a, args.host_b, args.remote_root
    has_rsync = shutil.which("rsync") is not None
    print("\n" + "=" * 78)
    print("Copy commands - run these from the project root on the LAPTOP")
    print("=" * 78)
    for label, host in (("SERVER A (1x 4090)", host_a), ("SERVER B (4x 3090)", host_b)):
        print(f"\n# --- {label} ---")
        print(f'ssh {user}@{host} "mkdir -p {remote}/data {remote}/results"')
        print(f'scp "{zip_path}" {user}@{host}:{remote}/')
        print(f'ssh {user}@{host} "cd {remote} && unzip -o satml2027_code.zip && rm satml2027_code.zip"')
        for name, path, _ in rows:
            destination = f"{remote}/data/" if name.startswith("data/") else f"{remote}/results/"
            if has_rsync:
                print(f'rsync -az --info=progress2 "{path}/" {user}@{host}:{destination}{Path(path).name}/')
            else:
                print(f'scp -r "{path}" {user}@{host}:{destination}')
    print("\n# verify the code bundle landed intact (both servers):")
    print(f'ssh {user}@{host_a} "cd {remote} && python3 -c \\"import pathlib;print(len(list(pathlib.Path(\'satml2027\').rglob(\'*.py\'))))\\""')
    print(f"# expected: {sum(1 for name in members if name.endswith('.py'))} python files")

    print("\n" + "=" * 78)
    print("Bring the results back - run these on the LAPTOP after Day 1 finishes")
    print("=" * 78)
    pull_root = (args.pull_root or (root / "results" / "satml2027" / "pull")).resolve()
    for tag, host in (("serverA", host_a), ("serverB", host_b)):
        local = (pull_root / tag).as_posix()
        print(f"\n# --- {tag} ---")
        if has_rsync:
            print(f'rsync -az --info=progress2 --exclude="*_features.npz" '
                  f'{user}@{host}:{remote}/results/satml2027/ "{local}/"')
        else:
            print(f'scp -r {user}@{host}:{remote}/results/satml2027 "{local}"')
    pull_a = (pull_root / "serverA").as_posix()
    pull_b = (pull_root / "serverB").as_posix()
    print("\n# merge both roots, then analyse (one command per registration):")
    for suffix, registration in (("A", "exp-20260920-019a.json"), ("B", "exp-20260920-019b.json")):
        experiment = f"EXP-20260920-019{suffix}"
        print(f"\npython satml2027/day1/d1_06_merge_verify.py "
              f'--registration configs/satml2027/{registration} '
              f'--roots "{pull_a}/{experiment}" "{pull_b}/{experiment}" '
              f"--out results/satml2027/merged/{experiment}")
        print(f"python satml2027/day1/d1_07_analyze.py "
              f"--merged results/satml2027/merged/{experiment} "
              f'--registration configs/satml2027/{registration} '
              f"--out results/satml2027/analysis/{experiment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
