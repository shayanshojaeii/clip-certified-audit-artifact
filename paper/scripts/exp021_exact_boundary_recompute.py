"""Exact-integer recomputation of the registered EXP-021 bootstrap tail probabilities (paper, Appendix E).

The registered analysis (``satml2027ext/analyze_ext.py``) forms each replicate of a family as
``point + 100 * sum_h f_h * (W_h - 1) @ (D_h / n)``: ``D`` holds each image's integer sum of paired outcome
differences (bank minus no correction) over its positions, ``n`` the family's positions, ``W_h`` the bootstrap
multiplicities of stratum ``h`` and ``f_h = sqrt(m_h / (m_h - 1))`` its Rao-Wu factor. It then counts replicates at
the test boundaries 0 and +-SESOI with inclusive comparisons. When an estimate lies exactly on a boundary, the
floating-point residue of ``D_h / n`` (about 1e-18) can put exactly tied replicates on either side.

This script repeats the registered draws (the same strata, seeds and PCG64DXSM streams, one call per stratum as in
the registered code), accumulates the integer numerators ``V_g = sum_{h in group g} (W_h - 1) @ D_h`` exactly (by
stratum size ``g``), and decides every comparison exactly: a replicate equals a boundary only if every ``V_g`` is zero
(each ``f_g`` is irrational and they are linearly independent over the rationals), and otherwise its sign is
resolved in floating point, or in 50-digit decimal arithmetic when the margin is small. It writes the exact raw tail
probabilities beside the released ones, with the Holm-adjusted direction and magnitude classes recomputed from the
exact values, and changes no released file.

usage (from the staged tree or the workspace):  python scripts/exp021_exact_boundary_recompute.py OUT_DIR
CPU only (numpy, scipy); about ten minutes.
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp021_replay_results as replay  # noqa: E402

FIELDS = ("p_two_sided", "tost_p_low", "tost_p_high", "tost_p_equivalence", "p_gain_at_least_sesoi",
          "p_loss_at_least_sesoi")
getcontext().prec = 50


def capture_analysis(key: str, registrations: dict) -> dict:
    """Run the registered analysis loader for 021A or 021B and capture its per-cell outcomes and image strata
    (the bootstrap itself is skipped here; it is recomputed exactly below)."""

    analyze_ext = replay.analyze_ext
    captured: dict = {"outcomes": {}, "rows": None, "images": None}
    original_outcomes, original_images, original_bootstrap = (analyze_ext.cell_outcomes, analyze_ext.build_images,
                                                              analyze_ext.rao_wu_bootstrap)

    def cell_outcomes(cell, data, **kwargs):
        computed = original_outcomes(cell, data, **kwargs)
        captured["outcomes"][cell["cell_id"]] = computed
        return computed

    def build_images(rows, keys):
        images = original_images(rows, keys)
        captured["rows"], captured["images"] = list(rows), images
        return images

    def no_bootstrap(aggregate, strata, *, replicates, seed, chunk=20000, return_unscaled=False):
        return {"deviation": np.zeros((2, aggregate.shape[1]))}

    analyze_ext.cell_outcomes, analyze_ext.build_images, analyze_ext.rao_wu_bootstrap = (
        cell_outcomes, build_images, no_bootstrap)
    try:
        replay.recompute(key, registrations)
    finally:
        analyze_ext.cell_outcomes, analyze_ext.build_images, analyze_ext.rao_wu_bootstrap = (
            original_outcomes, original_images, original_bootstrap)
    return captured


def exact_counts(n: int, totals: np.ndarray, groups: dict[int, np.ndarray], boundary: Fraction) -> tuple[np.ndarray, np.ndarray, int]:
    """Per contrast: replicates with value <= boundary and >= boundary (points), decided exactly; and how many
    comparisons needed 50-digit arithmetic."""

    sizes = sorted(groups)
    factors = {m: float(np.sqrt(m / (m - 1.0))) for m in sizes}
    first = groups[sizes[0]]
    replicates, width = first.shape
    s_float = np.zeros((replicates, width))
    zero = np.ones((replicates, width), dtype=bool)
    for m in sizes:
        s_float += factors[m] * groups[m]
        zero &= groups[m] == 0
    le = np.zeros(width, dtype=np.int64)
    ge = np.zeros(width, dtype=np.int64)
    precise = 0
    for j in range(width):
        target = boundary * n / 100 - int(totals[j])  # value <= boundary  <=>  S <= target (S = sum_g f_g V_g)
        t_float = float(target)
        diff = s_float[:, j] - t_float
        le_j = np.where(zero[:, j], Fraction(0) <= target, diff <= 0)
        ge_j = np.where(zero[:, j], Fraction(0) >= target, diff >= 0)
        close = np.flatnonzero((~zero[:, j]) & (np.abs(diff) < 1e-7))
        if close.size:
            t_dec = Decimal(target.numerator) / Decimal(target.denominator)
            for b in close:
                s_dec = sum(Decimal(m) .__truediv__(Decimal(m - 1)).sqrt() * Decimal(int(groups[m][b, j])) for m in sizes)
                le_j[b], ge_j[b] = s_dec <= t_dec, s_dec >= t_dec
            precise += int(close.size)
        le[j], ge[j] = int(le_j.sum()), int(ge_j.sum())
    return le, ge, precise


def recompute_family(name: str, cells: list[str], captured: dict, spec: dict, reference: str,
                     contrast_ids: list[str]) -> dict:
    analyze_ext = replay.analyze_ext
    rows, images = captured["rows"], captured["images"]
    positions = [i for i, row in enumerate(rows) if row["cell_id"] in set(cells)]
    order = {cell: k for k, cell in enumerate(cells)}
    positions.sort(key=lambda i: (order[rows[i]["cell_id"]], i))
    n = len(positions)
    width = len(contrast_ids)
    differences = np.zeros((n, width), dtype=np.int64)
    for out, index in enumerate(positions):
        row = rows[index]
        computed = captured["outcomes"][row["cell_id"]]
        key = f"standard@{computed['primary_key']}"
        reference_value = int(computed["outcomes"][reference][key][row["index"]])
        differences[out] = [int(computed["outcomes"][c][key][row["index"]]) - reference_value for c in contrast_ids]
    image_of = images["row_image"][positions]
    per_image = np.zeros((images["count"], width), dtype=np.int64)
    np.add.at(per_image, image_of, differences)
    totals = per_image.sum(axis=0)
    replicates = int(spec["replicates"])
    children = np.random.SeedSequence(spec["seed"]).spawn(len(images["strata"]))
    groups: dict[int, np.ndarray] = {}
    for stratum, child in zip(images["strata"], children):
        block = per_image[stratum["members"]]
        if not np.any(block):
            continue  # no image of this stratum in the family: no contribution (the stream is the stratum's own)
        size = int(stratum["size"])
        generator = np.random.Generator(np.random.PCG64DXSM(child))
        draws = generator.integers(0, size, size=(replicates, size), endpoint=False)  # as the registered code draws
        flat = (draws + (np.arange(replicates, dtype=np.int64) * size)[:, None]).ravel()
        counts = np.bincount(flat, minlength=replicates * size).reshape(replicates, size) - 1
        # integer-valued float64 product: exact while every partial sum stays below 2**53
        contribution = np.rint(counts.astype(np.float64) @ block.astype(np.float64)).astype(np.int64)
        if np.abs(contribution).max() > 2**50:
            raise RuntimeError("integer range exceeded")
        groups[size] = groups.get(size, 0) + contribution
    sesoi = Fraction(str(spec["sesoi_points"]))
    counts_at = {}
    precise = 0
    for label, boundary in (("zero", Fraction(0)), ("minus", -sesoi), ("plus", sesoi)):
        le, ge, extra = exact_counts(n, totals, groups, boundary)
        counts_at[label] = (le, ge)
        precise += extra
    exact = {}
    for j, contrast_id in enumerate(contrast_ids):
        le0, ge0 = counts_at["zero"][0][j], counts_at["zero"][1][j]
        p_low, p_high = counts_at["minus"][0][j] / replicates, counts_at["plus"][1][j] / replicates
        exact[contrast_id] = {
            "point_points": float(Fraction(int(totals[j]) * 100, n)),
            "p_two_sided": float(min(1.0, 2.0 * min(le0, ge0) / replicates)),
            "tost_p_low": float(p_low), "tost_p_high": float(p_high), "tost_p_equivalence": float(max(p_low, p_high)),
            "p_gain_at_least_sesoi": float(counts_at["plus"][0][j] / replicates),
            "p_loss_at_least_sesoi": float(counts_at["minus"][1][j] / replicates),
        }
    return {"n_positions": n, "exact": exact, "decimal_comparisons": precise}


def classes(entries: dict, primary: list[str], controls: list[str], alpha: float) -> dict:
    analyze_ext = replay.analyze_ext
    out = {}
    holm = {name: dict(zip(primary, analyze_ext.holm([entries[c][field] for c in primary])))
            for name, field in (("direction", "p_two_sided"), ("equivalence", "tost_p_equivalence"),
                                ("gain", "p_gain_at_least_sesoi"), ("loss", "p_loss_at_least_sesoi"))}
    for c in primary:
        out[c] = {"direction": analyze_ext.classify_direction(entries[c]["point_points"], holm["direction"][c], alpha),
                  "magnitude": analyze_ext.classify_magnitude(holm["gain"][c], holm["loss"][c], holm["equivalence"][c], alpha)}
    control_holm = dict(zip(controls, analyze_ext.holm([entries[c]["p_two_sided"] for c in controls])))
    for c in controls:
        out[c] = {"direction": analyze_ext.classify_direction(entries[c]["point_points"], control_holm[c], alpha)}
    return out


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    replay._load_replay()
    analyze_ext = replay.analyze_ext
    registrations = {key: replay.load_registration(path) for key, path in replay.REGISTRATION.items()}
    report: dict = {"method": __doc__.split("\n\n")[1].replace("\n", " "), "experiments": {}}
    lines = ["# Exact-boundary recomputation of the EXP-021 bootstrap tail probabilities", "",
             "Released (registered) value and exact value of every raw tail field that differs. All other fields are "
             "identical. Classes are the Holm-adjusted direction and magnitude classes recomputed from the exact values.", ""]
    total_fields = total_diff = class_changes = 0
    for key in ("A", "B"):
        registration = registrations[key]
        spec = analyze_ext.inference_spec(registration)
        captured = capture_analysis(key, registrations)
        released = replay.read_json(replay.ANALYSIS[key] / "analysis_ext.json")
        reference = spec["reference"]
        families: dict[str, list[str]] = {}
        for cell in registration["cells"]:
            if cell["cell_id"] in captured["outcomes"]:
                for name in analyze_ext._family_names(cell):
                    families.setdefault(name, []).append(cell["cell_id"])
        expected = [c for c in released["families"][next(iter(released["families"]))]["contrasts"]]
        experiment = {}
        for name, cells in sorted(families.items()):
            result = recompute_family(name, cells, captured, spec, reference, expected)
            rel_family = released["families"][name]["contrasts"]
            differences = []
            for contrast_id, values in result["exact"].items():
                rel = rel_family[contrast_id]
                if abs(rel["point_points"] - values["point_points"]) > 1e-9:
                    raise RuntimeError(f"{name}/{contrast_id}: point differs from the released output")
                for field in FIELDS:
                    total_fields += 1
                    if abs(rel[field] - values[field]) > 1e-12:
                        differences.append({"contrast": contrast_id, "field": field, "released": rel[field],
                                            "exact": values[field], "delta": values[field] - rel[field]})
            primary = [c for c in spec["primary_contrasts"] if c in result["exact"]]
            controls = [c for c in spec["control_contrasts"] if c in result["exact"]]
            exact_classes = classes(result["exact"], primary, controls, spec["family_alpha"])
            changed = []
            for contrast_id, cls in exact_classes.items():
                for field, value in cls.items():
                    if rel_family[contrast_id].get(field) not in (None, value):
                        changed.append({"contrast": contrast_id, "field": field,
                                        "released": rel_family[contrast_id][field], "exact": value})
            total_diff += len(differences)
            class_changes += len(changed)
            experiment[name] = {"n_positions": result["n_positions"], "differences": differences,
                                "class_changes": changed, "decimal_comparisons": result["decimal_comparisons"]}
            for d in differences:
                lines.append(f"- 021{key} `{name}`, `{d['contrast']}`, {d['field']}: released {d['released']:.5f}, "
                             f"exact {d['exact']:.5f}")
        report["experiments"][f"021{key}"] = experiment
    report["summary"] = {"raw_fields_compared": total_fields, "raw_fields_differing": total_diff,
                         "class_changes": class_changes}
    lines += ["", f"Fields compared: {total_fields}; differing: {total_diff}; direction or magnitude classes changed: "
              f"{class_changes}."]
    (out_dir / "exact_boundary_recomputation.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out_dir / "exact_boundary_recomputation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"fields compared {total_fields}, differing {total_diff}, class changes {class_changes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
