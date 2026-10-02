"""Copy every released file into the layout the code expects, then run the commands of REPRODUCE.md there.

usage (from the artifact root):  python paper/stage_inputs.py run

The artifact is organized by study (registrations/, results/exp017/, analysis/, theory/, control/, extension/,
followup/, paper/). The generators, the replay, the lint, the gate and the tests read their inputs at the paths of the
authors' workspace. ``paper/INPUT_MAP.json`` maps every released file to that path; this script copies each file to
``<target>/<workspace path>`` after checking its SHA-256 against ``MANIFEST.sha256.json``. Standard library only.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[1]


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = (ARTIFACT / sys.argv[1]).resolve() if not Path(sys.argv[1]).is_absolute() else Path(sys.argv[1])
    mapping = json.loads((ARTIFACT / "paper/INPUT_MAP.json").read_text(encoding="utf-8"))
    manifest = json.loads((ARTIFACT / "MANIFEST.sha256.json").read_text(encoding="utf-8"))
    problems = []
    for released, workspace_path in mapping.items():
        source = ARTIFACT / released
        data = source.read_bytes()
        expected = manifest.get(released, {}).get("sha256")
        if hashlib.sha256(data).hexdigest() != expected:
            problems.append(f"{released}: SHA-256 differs from MANIFEST.sha256.json")
            continue
        destination = target / workspace_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    if problems:
        print("\n".join(problems))
        print(f"STAGING FAILED: {len(problems)} file(s)")
        return 1
    print(f"staged {len(mapping)} files into {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
