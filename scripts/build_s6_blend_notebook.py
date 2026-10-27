#!/usr/bin/env python3
"""Build a Kaggle notebook that rank-blends one of our bundles with heliosli's public ConvNeXt-small model "S6" (or "R7").

S6 (October 13, 2026): four ConvNeXt-small folds on knee-centred 352 px views, trained on competition studies with
heliosli's own report labels, no OAI, no expert studies. Datasets heliosli/rsna-knee-compact-v1 (bundle
209a32c8b50a6b1c) and heliosli/rsna-orthofoundation-fold0-epoch7-20261004 (its code, pinned wheels and the
knee-finding model).
Stage 1 is our usual bundle notebook (a bundle = our exported weights, code and wheels), capped at 5.5 h (our 0.958
five folds, without mirrored views, need about 3 h on two T4s).
Stage 2 is S6's setup and run cells from heliosli's public notebook, unchanged except for output file names; they check
every file by SHA-256 and stop at 8.3 h from the notebook start, leaving unfinished studies unscored. S6 runs after our
stage, so the wheels it installs cannot affect ours. Stage 3 averages per-finding percentile ranks with weights fixed
before any leaderboard result (ours 0.7, S6 0.3), in plain Python so it does not depend on S6's packages. A study S6 did
not score takes our rank; if S6 scored fewer than 97% of the studies, or a stage fails, the notebook stops, so the score
always reflects the blend.

R7 (October 13, 2026, --model r7): heliosli's newer four-fold ConvNeXt-small (bundle 8c7b666e27dc0e8f, dataset
heliosli/rsna-knee-compact-r7cnxfsss), same views and code dataset, plus a fastMRI auxiliary task and two stackers fitted
on its own out-of-fold training predictions. Its setup and run cells come from heliosli's notebook
"0-957-latest-dataset-score-in-one-click" (cells 4 and 5), unchanged except for output names; the blend is the same with
R7 in S6's place. The 30% weight was chosen on 4,296 out-of-fold studies by a rule fixed beforehand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CREDIT_S6 = """# Knee: our CoAtNet five folds rank-blended with heliosli's public ConvNeXt-small (S6)

Stage 1 runs our own bundle. Stage 2 runs heliosli's compact ConvNeXt-small model "S6" as published in
[heliosli's notebook](https://www.kaggle.com/code/heliosli/rsna-knee-blend-gold-dataset-restored), with the public datasets
[heliosli/rsna-knee-compact-v1](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-v1) and
[heliosli/rsna-orthofoundation-fold0-epoch7-20261004](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004).
Stage 3 averages per-finding percentile ranks with fixed weights (ours 0.7, S6 0.3).
"""

CREDIT_R7 = """# Knee: our CoAtNet five folds rank-blended with heliosli's public ConvNeXt-small (R7)

Stage 1 runs our own bundle. Stage 2 runs heliosli's compact ConvNeXt-small model "R7" as published in
[heliosli's notebook](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click), with the public datasets
[heliosli/rsna-knee-compact-r7cnxfsss](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss) and
[heliosli/rsna-orthofoundation-fold0-epoch7-20261004](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004).
Stage 3 averages per-finding percentile ranks with fixed weights (ours 0.7, R7 0.3).
"""

SPECS = {
    "s6": {"bundle": "209a32c8b50a6b1c", "credit": CREDIT_S6, "receipt": "submission_receipt.json"},
    "r7": {"bundle": "8c7b666e27dc0e8f", "credit": CREDIT_R7, "receipt": "submission_receipt.json"},
}

START = """# Notebook clock (S6's 8.3 h deadline counts from here).
import time
NB_T0 = time.time()
"""

AFTER_OURS = """# Keep our predictions aside before S6 runs; stop if our stage did not produce real predictions.
import csv, os
from pathlib import Path
work = Path('/kaggle/working')
assert (work / 'submission.csv').is_file(), 'our stage wrote no submission.csv'
os.replace(work / 'submission.csv', work / 'ours_sub.csv')
with open(work / 'ours_sub.csv', newline='') as handle:
    _rows = list(csv.DictReader(handle))
if all(float(v) == 0.5 for r in _rows for k, v in r.items() if k != 'StudyInstanceUID'):
    raise RuntimeError('our stage produced only the 0.5 fallback')
OURS_SECONDS = time.time() - NB_T0
print('OURS_STAGE_DONE seconds %.0f' % OURS_SECONDS, flush=True)
"""

BLEND = """# Stage 3: per-finding rank blend with weights fixed before any leaderboard result: ours 0.7, S6 0.3 (plain Python).
import csv, json, math, time
from pathlib import Path
W_OURS, W_S6 = 0.7, 0.3
work = Path('/kaggle/working')
with open(DATA / 'test.csv', newline='') as handle:
    test_ids = [row['StudyInstanceUID'] for row in csv.DictReader(handle)]


def read(path):
    with open(path, newline='') as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, {row['StudyInstanceUID']: row for row in reader}


columns, ours = read(work / 'ours_sub.csv')
labels = [c for c in columns if c != 'StudyInstanceUID']
assert labels == list(bundle_metadata['targets']), 'our columns differ from the bundle targets'
_, s6 = read(work / 's6_sub.csv')
for rows, name in ((ours, 'ours'), (s6, 's6')):
    assert sorted(rows) == sorted(test_ids), name + ': rows differ from test.csv'
    assert all(label in next(iter(rows.values())) for label in labels), name + ': columns differ'
scored = set()  # S6 fills a study it did not score with the column mean: read what its workers actually scored
for path in pred.glob('worker*/predictions.csv'):
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            try:
                values = [float(row[label]) for label in labels]
            except (KeyError, ValueError, TypeError):
                continue
            if all(0.0 <= v <= 1.0 for v in values):
                scored.add(row['StudyInstanceUID'])
s6_failed = [u not in scored for u in test_ids]
n = len(test_ids)
if sum(s6_failed) > 0.03 * n:
    raise RuntimeError('S6 did not score %d of %d studies' % (sum(s6_failed), n))


def pct(values):
    # percentile ranks with ties averaged, as pandas' rank(method='average', pct=True)
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j + 2) / 2 / len(values)
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = pct(a), pct(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return round(num / den, 4) if den > 0 else None


blended, agreement = {}, {}
for label in labels:
    mine = pct([float(ours[u][label]) for u in test_ids])
    theirs = pct([float(s6[u][label]) for u in test_ids])
    blended[label] = [m if failed else W_OURS * m + W_S6 * t for m, t, failed in zip(mine, theirs, s6_failed)]
    if n >= 3:
        agreement[label] = spearman([float(ours[u][label]) for u in test_ids], [float(s6[u][label]) for u in test_ids])
with open(work / 'submission.csv', 'w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(columns)
    for i, u in enumerate(test_ids):
        values = [blended[label][i] for label in labels]
        assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values)
        writer.writerow([u] + ['%.6f' % v for v in values])
receipt = {'weights': {'ours': W_OURS, 's6': W_S6}, 'rows': n, 's6_failed': sum(s6_failed),
           'ours_seconds': round(OURS_SECONDS, 1), 's6_seconds': round(time.time() - NB_T0 - OURS_SECONDS, 1),
           'total_seconds': round(time.time() - NB_T0, 1), 'spearman_ours_vs_s6': agreement}
(work / 'blend_receipt.json').write_text(json.dumps(receipt, indent=1))
print('NOTEBOOK_COMPLETE', json.dumps(receipt), flush=True)
"""


def cell(kind: str, text: str) -> dict:
    item = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def replace_once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one {what}: {old!r}")
    return text.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(SPECS), default="s6")
    parser.add_argument("--s6-notebook", "--public-notebook", dest="public_notebook", type=Path, required=True,
                        help="heliosli's notebook whose cells 4 and 5 run the model (S6: rsna-knee-blend-gold-dataset-restored; "
                             "R7: 0-957-latest-dataset-score-in-one-click)")
    parser.add_argument("--our-notebook", type=Path, required=True, help="Our bundle notebook (one code cell)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec, m, M = SPECS[args.model], args.model, args.model.upper()
    public = json.loads(args.public_notebook.read_text())
    setup, run = ("".join(public["cells"][i]["source"]) for i in (4, 5))
    if f"BUNDLE_ID = '{spec['bundle']}'" not in setup or "T0 = NB_T0" not in setup or "DEADLINE_HOURS = 8.3" not in setup:
        raise ValueError(f"Unexpected {M} setup cell: expected bundle {spec['bundle']}, T0 = NB_T0 and the 8.3 h deadline")
    if "probe_dual.py" not in run:
        raise ValueError(f"Unexpected {M} run cell")
    run = replace_once(run, "import pandas as pd\ndisplay(pd.read_csv(WORK / 'ours_sub.csv').head())", "", "display")
    run = replace_once(run, "shutil.copyfile(pred / 'submission.csv', WORK / 'ours_sub.csv')",
                       f"shutil.copyfile(pred / 'submission.csv', WORK / '{m}_sub.csv')", f"{M} output copy")
    run = replace_once(run, "pd.read_csv(WORK / 'ours_sub.csv')", f"pd.read_csv(WORK / '{m}_sub.csv')", "probe read")
    run = replace_once(run, "_df.to_csv(WORK / 'ours_sub.csv', index=False)", f"_df.to_csv(WORK / '{m}_sub.csv', index=False)",
                       "probe write")
    run = replace_once(run, f"(WORK / '{spec['receipt']}').write_text", f"(WORK / '{m}_receipt.json').write_text", "receipt")
    run = replace_once(run, "print('OURS_STAGE_COMPLETE'", f"print('{M}_STAGE_COMPLETE'", "stage marker")
    if "ours_sub.csv" in run:
        raise ValueError(f"{M} run cell still writes ours_sub.csv")
    ours_notebook = json.loads(args.our_notebook.read_text())
    if len(ours_notebook["cells"]) != 1:
        raise ValueError("Expected our bundle notebook to have one code cell")
    ours = replace_once("".join(ours_notebook["cells"][0]["source"]), "deadline_seconds=8 * 3600",
                        "deadline_seconds=int(5.5 * 3600)", "time limit")
    notebook = dict(ours_notebook)
    blend = BLEND if m == "s6" else BLEND.replace("s6", m).replace("S6", M)
    notebook["cells"] = [cell("markdown", spec["credit"]), cell("code", START), cell("code", ours), cell("code", AFTER_OURS),
                         cell("code", setup), cell("code", run), cell("code", blend)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1))
    print(f"wrote {args.output}: {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
