"""Shared paths and re-exports of the frozen hashing helpers.

The hashing scheme is imported from ``satml2027/common/hashing.py`` so that every
item-list hash and registration hash produced here is computed by exactly the
same function as the EXP-019 registrations.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SATML_DIR = PROJECT_ROOT / "satml2027"
if str(SATML_DIR) not in sys.path:
    sys.path.insert(0, str(SATML_DIR))

import json  # noqa: E402
import math  # noqa: E402

from common.hashing import (  # noqa: E402
    canonical_json_hash,
    sha256_file,
    sha256_id_list,
    write_json_atomic,
)


class DuplicateKeyError(ValueError):
    """A JSON object repeats a key; EXP-021 refuses such files (V3, re-review n3)."""


def _refuse_duplicate_keys(pairs):
    seen: dict = {}
    for key, value in pairs:
        if key in seen:
            raise DuplicateKeyError(f"duplicate JSON key {key!r}")
        seen[key] = value
    return seen


class NonFiniteJSONError(ValueError):
    """A JSON file carries NaN or Infinity; EXP-021 refuses such files (V4, reviewer-bundle V3 audit E2)."""


def _refuse_constant(token):
    raise NonFiniteJSONError(f"non-finite JSON number {token!r}")


def _finite_float(token):
    value = float(token)
    if not math.isfinite(value):  # V4.1: an overflowing literal such as 1e999 parses to inf
        raise NonFiniteJSONError(f"non-finite JSON number {token!r}")
    return value


def loads_json(text):
    """Parse JSON text with the strict EXP-021 rules (see ``read_json``)."""

    return json.loads(text, object_pairs_hook=_refuse_duplicate_keys, parse_constant=_refuse_constant,
                      parse_float=_finite_float)


def read_json(path):
    """Read UTF-8 JSON and refuse any object with a repeated key or a non-finite number.

    The frozen ``common.hashing.read_json`` keeps the last value of a repeated
    key, so an approval with ``"sampling_authorized": false`` followed by
    ``"sampling_authorized": true`` would pass; every EXP-021 reader uses this
    strict version instead.  V4 (audit E2): the tokens ``NaN``, ``Infinity`` and
    ``-Infinity``, which Python's parser accepts by default, are refused too; the
    frozen writers already refuse to serialize them (``allow_nan=False``).
    V4.1: a finite-looking literal that overflows to infinity (``1e999``) is
    refused as well.
    """

    return loads_json(Path(path).read_text(encoding="utf-8"))

__all__ = [
    "DuplicateKeyError",
    "NonFiniteJSONError",
    "PROJECT_ROOT",
    "SATML_DIR",
    "canonical_json_hash",
    "loads_json",
    "read_json",
    "sha256_file",
    "sha256_id_list",
    "write_json_atomic",
]
