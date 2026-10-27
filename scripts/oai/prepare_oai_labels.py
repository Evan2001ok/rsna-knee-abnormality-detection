#!/usr/bin/env python3
"""Labels for OAI knees from their MOAKS MRI gradings, mapped to the challenge's twelve findings ("v1" below:
made without --fixed).

The mapping follows the host's definitions (forum topic 733343); findings MOAKS does not grade (MCL, contusion, fracture)
and items missing for a reading stay blank, so training skips them. Where MOAKS cannot tell the host's threshold apart
(e.g. a partial ACL tear may or may not be high grade), the label is an in-between value. Several reading projects can
grade the same knee and visit; their labels are averaged.

- ACL: complete tear 1, partial tear 0.5, none 0; a reconstructed ACL is left blank.
- Medial / Lateral Meniscus: any tear type (radial, horizontal, vertical, complex, maceration) in any horn or body, or a
  root tear, is 1; intrasubstance signal only (does not reach the surface) or normal is 0.
- Medial / Lateral / PF OA: worst subregion of the compartment (medial: femur central/posterior and tibia; lateral:
  likewise; patellofemoral: patella and trochlea, the femoral anterior subregions). Full-thickness loss over 10% or more
  of a subregion 1; any full-thickness loss under 10% 0.7; partial-thickness loss only, over 10% or more 0.35;
  partial-thickness loss only, under 10% 0.1; none 0. The host asks for a moderate or large area of more than 50% thickness loss.
- Effusion: MOAKS effusion-synovitis moderate or severe 1, mild 0.1, none 0 (host: moderate or large).
- Synovitis: MOAKS Hoffa synovitis severe 1, moderate 0.7, mild 0.2, none 0 (host: no size threshold).
- Baker's: popliteal cyst present 0.6 (MOAKS records no size; the host asks for moderate or large), absent 0.
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee_abnormality_detection.constants import TARGETS
from rsna_knee_abnormality_detection.utils import file_hash, write_json

MOAKS_DIR = Path("data/oai/assessments/Semi-Quant Scoring_ASCII")
COMPARTMENTS = {
    "Medial OA": ["FMC", "FMP", "TMA", "TMC", "TMP"],
    "Lateral OA": ["FLC", "FLP", "TLA", "TLC", "TLP"],
    "PF OA": ["PM", "PL", "FMA", "FLA"],
}
TEAR_CODES = {2, 3, 4, 5, 6, 7, 8}  # MOAKS meniscal morphology: 0 normal, 1 signal only, 2-8 tears and maceration


def code(series: pd.Series) -> pd.Series:
    """Leading number of a MOAKS value such as '2.1: 10-75% area...'; missing ('.: Missing ...') becomes NaN."""
    return pd.to_numeric(series.astype(str).str.extract(r"^\s*(\d+(?:\.\d+)?)", expand=False), errors="coerce")


def cartilage_label(value: pd.Series, fixed: bool = False) -> pd.Series:
    """MOAKS cartilage 'A.B': A = area of any loss, B = area of full-thickness loss (0 none, 1 <10%, 2 10-75%, 3 >75%).

    fixed (Oct 17, 2026): follow-up reads also use the literal code 0.5 / -0.5 (MOAKS: the same grade as the previous
    visit, with a definite worsening / improvement); v1 read 0.5 as area 0 with full-thickness 5 and labelled it 1.0.
    With fixed, such cells are blank here, and visit_labels blanks the whole compartment unless another subregion
    already gives 1.0."""
    area = np.floor(value)
    full = np.round((value - area) * 10)
    label = pd.Series(np.nan, index=value.index)
    if fixed:
        value = value.where(full <= 3)
        area, full = np.floor(value), np.round((value - np.floor(value)) * 10)
    label[area == 0] = 0.0
    label[area == 1] = 0.1
    label[(area >= 2) & (full == 0)] = 0.35
    label[(area >= 2) & (full == 1)] = 0.7
    label[full >= 2] = 1.0
    label[(area == 1) & (full >= 1)] = 0.7
    return label


def has_full_thickness_detail(frame: pd.DataFrame, visit: str) -> bool:
    """Whether a visit's file records full-thickness areas (a cartilage code with a decimal digit 1-3); our V01 file
    holds whole-number codes only, which v1 read as 'no full-thickness loss' everywhere."""
    columns = [c for c in frame.columns if re.fullmatch(rf"V{visit}MCM[A-Z]+", c) and not c.endswith("NTS")]
    values = pd.concat([code(frame[c]) for c in columns]).dropna() if columns else pd.Series(dtype=float)
    digits = np.round((values - np.floor(values)) * 10)
    return bool(((digits >= 1) & (digits <= 3)).any())


def visit_labels(frame: pd.DataFrame, visit: str, fixed: bool = False) -> pd.DataFrame:
    col = lambda stem: frame.get(f"V{visit}M{stem}", pd.Series(np.nan, index=frame.index))  # noqa: E731
    out = pd.DataFrame(index=frame.index)
    acl = code(col("ACLTR")).map({0: 0.0, 1: 0.5, 2: 1.0})
    acl[code(col("ACLRP")) == 1] = np.nan
    out["ACL"] = acl
    for name, side in (("Medial Meniscus", "M"), ("Lateral Meniscus", "L")):
        horns = pd.concat([code(col(f"MT{side}{part}")) for part in "ABP"], axis=1)
        root = code(col(f"MRT{side}"))
        torn = horns.isin(TEAR_CODES).any(axis=1) | (root == 1)
        known = horns.notna().any(axis=1) | root.notna()
        out[name] = np.where(known, torn.astype(float), np.nan)
    blank_oa = fixed and not has_full_thickness_detail(frame, visit)
    for name, regions in COMPARTMENTS.items():
        out[name] = pd.concat([cartilage_label(code(col(f"CM{r}")), fixed) for r in regions], axis=1).max(axis=1, skipna=True)
        if fixed:
            # A within-grade code means the subregion is at least as bad as at the previous visit, which this row does
            # not show; the worst of the other subregions would understate the compartment, so it is blank unless
            # another subregion already gives the top label.
            within = pd.concat([col(f"CM{r}").astype(str).str.match(r"\s*-?0\.5(?!\d)") for r in regions], axis=1).any(axis=1)
            out.loc[within & ~(out[name] >= 1.0), name] = np.nan
        if blank_oa:
            out[name] = np.nan
    out["Effusion"] = code(col("EFFWK")).map({0: 0.0, 1: 0.1, 2: 1.0, 3: 1.0})
    out["Synovitis"] = code(col("SYIC")).map({0: 0.0, 1: 0.2, 2: 0.7, 3: 1.0})
    out["Baker's"] = code(col("POPCYS")).map({0: 0.0, 1: 0.6})
    for name in ("MCL", "Contusion", "Fracture"):
        out[name] = np.nan
    out["oai_id"] = frame["ID"].astype(str)
    out["side"] = code(frame["SIDE"]).map({1: "RIGHT", 2: "LEFT"})  # "1: Right" in most files, plain "1" in visit 01
    out["visit"] = f"V{visit}"
    out["read_project"] = frame.get("READPRJ", pd.Series("", index=frame.index)).astype(str)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/labels/oai_moaks_v1"))
    parser.add_argument("--fixed", action="store_true",
                        help="Oct 17, 2026: blank the OA findings where a subregion has the within-grade code 0.5 / -0.5 "
                             "(unless another subregion gives 1.0) and for visit files without full-thickness detail "
                             "(V01); v1 otherwise")
    args = parser.parse_args()
    reads, inputs = [], {}
    for path in sorted(glob.glob(str(MOAKS_DIR / "kMRI_SQ_MOAKS_BICL*.txt"))):
        visit = re.search(r"BICL(\d\d)", path).group(1)
        frame = pd.read_csv(path, sep="|", low_memory=False, encoding="latin-1")
        frame.columns = [c.upper() if c.lower() in {"id", "side", "version", "readprj"} else c for c in frame.columns]
        reads.append(visit_labels(frame, visit, args.fixed))
        inputs[path] = file_hash(Path(path))
    reads = pd.concat(reads, ignore_index=True)
    labels = reads.groupby(["oai_id", "side", "visit"])[TARGETS].mean().reset_index()
    labels.insert(0, "StudyInstanceUID", "OAI_" + labels["oai_id"] + "_" + labels["side"] + "_" + labels["visit"])
    values = labels[TARGETS].to_numpy(dtype=float)
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Labels must be between zero and one")
    labels = labels[labels[TARGETS].notna().any(axis=1)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "labels.csv"
    labels.to_csv(output, index=False)
    summary = {name: {"known": int(labels[name].notna().sum()), "mean": round(float(labels[name].mean()), 4)
                      if labels[name].notna().any() else None,
                      "share_at_least_half": round(float((labels[name] >= 0.5).mean()), 4)} for name in TARGETS}
    write_json(args.output_dir / "label_provenance.json", {
        "rule": __doc__.split("\n\n", 1)[1].strip(), "fixed": args.fixed, "inputs": inputs, "knee_visits": len(labels),
        "distinct_knees": int(labels[["oai_id", "side"]].drop_duplicates().shape[0]), "per_finding": summary,
        "output_sha256": file_hash(output)})
    print(f"wrote {output}: {len(labels)} knee-visits")
    for name, s in summary.items():
        print(f"  {name:17} known {s['known']:6}  mean {s['mean']}  >=0.5 {s['share_at_least_half']}")


if __name__ == "__main__":
    main()
