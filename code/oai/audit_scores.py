"""Audit OAI raw scores and link baseline knee records without inventing labels."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_nda(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        next(reader)  # NDA column descriptions, not a participant record.
        return list(reader)


def main():
    paths = list((ROOT / "scores").rglob("oai_kmrisemiquant01*.txt"))
    if len(paths) != 1:
        raise SystemExit(f"Expected one scoring table, found {len(paths)}. Download package 1249833 first.")
    scores = read_nda(paths[0])
    images = read_nda(ROOT / "baseline" / "image03.txt")
    with (ROOT / "score_dictionary.csv").open(encoding="utf-8-sig", newline="") as f:
        definitions = {r["ElementName"]: r for r in csv.DictReader(f)}
    with (ROOT / "tool_state/NDA/nda-tools/downloadcmd/packages/1249823/package_file_metadata_1249823.txt").open(encoding="utf-8-sig", newline="") as f:
        sizes = {r["NDA_S3_URL"]: int(r["FILE_SIZE"]) for r in csv.DictReader(f)}
    baseline_values = {"v00", "00", "0", "baseline"}
    base_scores = [r for r in scores if r.get("visit", "").strip().lower() in baseline_values]
    candidates = defaultdict(list)
    for row in images:
        desc = row["image_description"]
        if row["image_modality"] == "MRI" and desc.startswith("SAG_IW_TSE_"):
            side = desc.rsplit("_", 1)[1]
            if side in {"LEFT", "RIGHT"}:
                candidates[(row["src_subject_id"].strip(), side)].append(row)
    score_groups = defaultdict(list)
    for row in base_scores:
        side = {"1": "RIGHT", "2": "LEFT"}.get(row.get("side", "").strip())
        if side:
            score_groups[(row["src_subject_id"].strip(), side)].append(row)
    matched = set(score_groups) & set(candidates)
    fields = [k for k in scores[0] if k.startswith(("m_", "w_", "b_"))]
    field_audit = {}
    for field in fields:
        d = definitions.get(field, {})
        present = [r for r in base_scores if r.get(field, "").strip()]
        field_audit[field] = {
            "description": d.get("ElementDescription", ""),
            "value_range": d.get("ValueRange", ""),
            "notes": d.get("Notes", ""),
            "baseline_nonempty_rows": len(present),
            "baseline_raw_values": dict(Counter(r[field] for r in present)),
            "all_visit_raw_values": dict(Counter(r[field] for r in scores if r.get(field, "").strip())),
            "baseline_matched_knees": sum(any(r.get(field, "").strip() for r in score_groups[k]) for k in matched),
        }
    manifest = []
    for key in sorted(matched):
        for image in candidates[key]:
            manifest.append({"src_subject_id": key[0], "side": key[1], "image_file": image["image_file"],
                             "bytes": sizes[image["image_file"]], "duplicate_series": len(candidates[key]),
                             "score_rows": len(score_groups[key]),
                             "read_projects": ";".join(sorted({r.get("readprj", "") for r in score_groups[key]}))})
    out = ROOT / "score_audit"
    out.mkdir(exist_ok=True)
    with (out / "baseline_sag_iw_candidates.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["src_subject_id", "side", "image_file", "bytes", "duplicate_series", "score_rows", "read_projects"])
        writer.writeheader()
        writer.writerows(manifest)
    summary = {
        "score_file": str(paths[0]), "rows": len(scores),
        "participants": len({r["src_subject_id"] for r in scores}),
        "visits": dict(Counter(r.get("visit", "") for r in scores)),
        "projects": dict(Counter(r.get("readprj", "") for r in scores)),
        "versions": dict(Counter(r.get("version", "") for r in scores)),
        "sides": dict(Counter(r.get("side", "") for r in scores)),
        "baseline_rows": len(base_scores), "baseline_unilateral_knees": len(score_groups),
        "baseline_matched_sag_iw_knees": len(matched),
        "unmatched_baseline_knees": [list(k) for k in sorted(set(score_groups) - set(candidates))],
        "candidate_series": len(manifest), "candidate_GB": sum(r["bytes"] for r in manifest) / 1e9,
        "duplicate_score_knees": sum(len(score_groups[k]) > 1 for k in matched),
        "duplicate_image_knees": sum(len(candidates[k]) > 1 for k in matched),
        "label_status": "Raw scores only. Missing values remain unknown. Thresholds and reader/project precedence require codebook review.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "field_coverage.json").write_text(json.dumps(field_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
