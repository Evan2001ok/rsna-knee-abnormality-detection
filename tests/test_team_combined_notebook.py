"""The combined team notebook's own logic (scripts/build_team_combined_notebook.py), run locally on fake files: the
time-limited stage for the teammates' sets and the final blend, including the fallbacks when a part fails.

The blend has three parts: "theirs" (the teammates' model sets), "ours" (our models) and "r7" (R7, heliosli's public
ConvNeXt-small, named after its weights dataset). The tests that build the whole notebook are skipped unless the two
source notebooks (the teammates' notebook and our earlier Kaggle notebook) are present; neither is in this
repository."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("builder", Path(__file__).parents[1] / "scripts/build_team_combined_notebook.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
           "Synovitis", "Baker's", "Contusion", "Fracture"]


def localise(code: str, root: Path) -> str:
    return (code.replace("/kaggle/working", str(root / "working")).replace("/kaggle/temp", str(root / "temp"))
            .replace("/kaggle/input/competitions/rsna-knee-abnormality-detection", str(root / "data")))


def write_frame(path: Path, ids: list[str], values: np.ndarray) -> None:
    frame = pd.DataFrame(values, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


@pytest.fixture
def setup(tmp_path):
    ids = [f"1.2.{i}" for i in range(40)]
    rng = np.random.default_rng(0)
    (tmp_path / "data").mkdir()
    write_frame(tmp_path / "data/sample_submission.csv", ids, np.full((40, 12), 0.5))
    parts = {name: rng.random((40, 12)) for name in ("theirs", "ours", "r7")}
    for name, values in parts.items():
        write_frame(tmp_path / f"working/{name}_sub.csv", ids, values)
    return tmp_path, ids, parts


def run_blend(root: Path, ids: list[str], ok: dict, r7_scored: list[str]) -> tuple[pd.DataFrame, dict]:
    pred = root / "temp/probe/prediction"
    write_frame(pred / "worker0/predictions.csv", r7_scored, np.full((len(r7_scored), 12), 0.3))
    env = {"THEIRS_OK": ok["theirs"], "OURS_OK": ok["ours"], "R7_OK": ok["r7"], "pred": pred, "NB_T0": time.time(),
           "THEIRS_DONE": ["dino", "eff"], "THEIRS_SECONDS": 1.0, "OURS_SECONDS": 2.0}
    exec(compile(localise(builder.BLEND, root), "blend", "exec"), env)
    out = pd.read_csv(root / "working/submission.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    return out, json.loads((root / "working/blend_receipt.json").read_text())


def expected(parts: dict, weights: dict, ids: list[str]) -> pd.DataFrame:
    ranks = {k: pd.DataFrame(v, index=ids, columns=TARGETS).rank(method="average", pct=True) for k, v in parts.items()}
    return sum(w * ranks[k] for k, w in weights.items()) / sum(weights.values())


def test_all_parts(setup):
    root, ids, parts = setup
    out, receipt = run_blend(root, ids, {"theirs": True, "ours": True, "r7": True}, ids)
    want = expected(parts, {"theirs": 0.5, "ours": 0.3, "r7": 0.2}, ids)
    assert np.abs(out.loc[ids, TARGETS].to_numpy() - want.to_numpy()).max() < 1e-6
    assert list(out.columns) == TARGETS and receipt["left_out"] == {}


def test_r7_failed_rescales_the_rest(setup):
    root, ids, parts = setup
    out, receipt = run_blend(root, ids, {"theirs": True, "ours": True, "r7": False}, ids)
    want = expected({k: parts[k] for k in ("theirs", "ours")}, {"theirs": 0.5, "ours": 0.3}, ids)
    assert np.abs(out.loc[ids, TARGETS].to_numpy() - want.to_numpy()).max() < 1e-6
    assert receipt["left_out"] == {"r7": "stage failed"}


def test_r7_partly_scored(setup):
    root, ids, parts = setup
    out, receipt = run_blend(root, ids, {"theirs": True, "ours": True, "r7": True}, ids[:39])  # 1 of 40 = 2.5%
    full = expected(parts, {"theirs": 0.5, "ours": 0.3, "r7": 0.2}, ids)
    two = expected({k: parts[k] for k in ("theirs", "ours")}, {"theirs": 0.5, "ours": 0.3}, ids)
    assert np.abs(out.loc[ids[:39], TARGETS].to_numpy() - full.loc[ids[:39]].to_numpy()).max() < 1e-6
    assert np.abs(out.loc[ids[39:], TARGETS].to_numpy() - two.loc[ids[39:]].to_numpy()).max() < 1e-6
    assert receipt["r7_not_scored"] == 1
    out, receipt = run_blend(root, ids, {"theirs": True, "ours": True, "r7": True}, ids[:36])  # 10%: left out
    assert "r7" not in receipt["weights_used"] and receipt["left_out"]["r7"].startswith("left out")


def test_only_one_part_and_none(setup):
    root, ids, parts = setup
    out, _ = run_blend(root, ids, {"theirs": False, "ours": True, "r7": False}, ids)
    want = expected({"ours": parts["ours"]}, {"ours": 1.0}, ids)
    assert np.abs(out.loc[ids, TARGETS].to_numpy() - want.to_numpy()).max() < 1e-6
    with pytest.raises(RuntimeError, match="no stage produced"):
        run_blend(root, ids, {"theirs": False, "ours": False, "r7": False}, ids)


DUMMY = '''
import os, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
out = Path(os.environ["TEAM_OUTPUT_DIR"]) / "six_model_predictions"
ids = pd.read_csv(sys.argv[0].replace("working/theirs/team_sets.py", "data/sample_submission.csv"), dtype=str)["StudyInstanceUID"]
TARGETS = __TARGETS__
Path(sys.argv[0]).parents[2].joinpath("temp/rsna_six_model_shared384_v1").mkdir(parents=True, exist_ok=True)
for i, key in enumerate(("dino", "eff")):
    d = out / key
    d.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(np.full((len(ids), 12), 0.2 + 0.4 * i), columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", ids)
    frame.to_csv(d / "test_probabilities.csv", index=False)
    (d / "complete.json").write_text("{}")
(out / "dino3").mkdir(parents=True, exist_ok=True)
(Path(os.environ["TEAM_OUTPUT_DIR"]).parent / "submission.csv").write_text("stray")
time.sleep(60)  # the third set never finishes
'''


def test_theirs_stage_time_limit_keeps_finished_sets(setup):
    root, ids, _ = setup
    (root / "working/theirs_sub.csv").unlink()
    code = builder.THEIRS_STAGE.replace("__SETS__", repr(builder.SETS)).replace("__LIMIT__", "8").replace(
        "__SOURCE__", repr(DUMMY.replace("__TARGETS__", repr(TARGETS))))
    env = {"NB_T0": time.time()}
    start = time.time()
    exec(compile(localise(code, root), "theirs", "exec"), env)
    assert time.time() - start < 30, "the time limit did not stop the process"
    assert env["THEIRS_OK"] and env["THEIRS_DONE"] == ["dino", "eff"] and env["THEIRS_RC"] == "time limit"
    got = pd.read_csv(root / "working/theirs_sub.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    assert sorted(got.index) == sorted(ids) and np.allclose(got[TARGETS].to_numpy(), 0.4)  # mean of 0.2 and 0.6
    assert not (root / "temp/rsna_six_model_shared384_v1").exists(), "their image store must be deleted"
    assert not (root / "working/submission.csv").exists(), "a stray submission.csv must be removed"


def test_built_notebook_compiles(tmp_path):
    if not Path("notebooks/rsna-ensemble.ipynb").is_file() or not Path(
            "artifacts/kaggle-oaiseeds6/notebook/inference.ipynb").is_file():
        pytest.skip("source notebooks not present")
    sys.argv = ["build", "--output-dir", str(tmp_path)]
    builder.main()
    nb = json.loads((tmp_path / "inference.ipynb").read_text())
    assert len(nb["cells"]) == 6
    import ast
    cell = ast.parse("".join(nb["cells"][2]["source"]))
    embedded = [n.value for n in ast.walk(cell) if isinstance(n, ast.Constant) and isinstance(n.value, str) and "_KEEP" in n.value]
    assert len(embedded) == 1
    src = embedded[0]
    assert "_KEEP = ('dino', 'eff', 'dino3')" in src and "TEAM_OUTPUT_DIR" in src and "'eff': (0.0, 12.0)," in src
    compile(src, "team_sets", "exec")


# Stage 4 with --postproc noisyor-sibling (Oct 17, 2026): a noisy-OR blend of ranks, then fixed corrections between
# related findings ("siblings"); docs/WRITEUP.md describes it.
SIBLINGS = {"Lateral OA": ("Medial OA", -0.10), "PF OA": ("Medial OA", -0.10), "Synovitis": ("Effusion", -0.20)}
W = {"theirs": 0.5, "ours": 0.3, "r7": 0.2}


def reference(parts: dict, weights: dict, failed: dict | None = None, siblings: dict = SIBLINGS) -> np.ndarray:
    """The formulas of our earlier out-of-fold check of this method (the "verifier", a script not included here:
    noisy_or, then sibling with -0.10/-0.10/-0.20) in numpy, extended the way the notebook handles failures: a part is
    ranked among the studies it did not fail on and a study takes the weighted mean of the parts that did not fail on
    it."""
    from scipy.stats import norm, rankdata
    n = len(next(iter(parts.values())))
    failed = failed or {}
    num, den = np.zeros((n, 12)), np.zeros(n)
    for name, values in parts.items():
        ok = ~failed.get(name, np.zeros(n, bool))
        m = ok.sum()
        term = np.zeros((n, 12))
        term[ok] = -np.log(1 - rankdata(values[ok], axis=0) / m + 0.5 / m)
        num += weights[name] * term
        den += weights[name] * ok
    with np.errstate(invalid="ignore"):
        blended = num / den[:, None]
    blended[den == 0] = np.nanmedian(blended, axis=0)
    z = norm.ppf(rankdata(blended, axis=0) / n - 0.5 / n)
    out = z.copy()
    for label, (partner, weight) in siblings.items():
        out[:, TARGETS.index(label)] = z[:, TARGETS.index(label)] + weight * z[:, TARGETS.index(partner)]
    return rankdata(out, axis=0) / n


def correlated(n: int, seed: int) -> dict:
    """Three parts that share a signal, with related findings linked, so the ranks are not trivial."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 12))
    base[:, TARGETS.index("Lateral OA")] += 0.6 * base[:, TARGETS.index("Medial OA")]
    base[:, TARGETS.index("Synovitis")] += 0.8 * base[:, TARGETS.index("Effusion")]
    return {name: 1 / (1 + np.exp(-(base + rng.normal(scale=s, size=(n, 12)))))
            for name, s in (("theirs", 0.5), ("ours", 0.7), ("r7", 1.5))}


def run_noisyor(root: Path, ids: list[str], parts: dict, ok: dict | None = None, r7_scored: list[str] | None = None,
                ours_filled: list[str] | None = None, theirs_empty: list[str] | None = None,
                code: str | None = None) -> tuple[pd.DataFrame, dict]:
    (root / "data").mkdir(parents=True, exist_ok=True)
    write_frame(root / "data/sample_submission.csv", ids, np.full((len(ids), 12), 0.5))
    for name, values in parts.items():
        write_frame(root / f"working/{name}_sub.csv", ids, values)
    r7_scored = ids if r7_scored is None else r7_scored
    pred = root / "temp/probe/prediction"
    write_frame(pred / "worker0/predictions.csv", r7_scored, np.full((len(r7_scored), 12), 0.3))
    if ours_filled is not None:
        (root / "working/inference-summary.json").write_text(json.dumps({"filled_study_ids": ours_filled}))
    if theirs_empty is not None:
        (root / "working/theirs").mkdir(parents=True, exist_ok=True)
        (root / "working/theirs/no_series_studies.json").write_text(json.dumps(theirs_empty))
    ok = ok or {"theirs": True, "ours": True, "r7": True}
    env = {"THEIRS_OK": ok["theirs"], "OURS_OK": ok["ours"], "R7_OK": ok["r7"], "pred": pred, "NB_T0": time.time(),
           "THEIRS_DONE": ["dino", "eff"], "THEIRS_SECONDS": 1.0, "OURS_SECONDS": 2.0}
    exec(compile(localise(code or builder.BLEND_NOISYOR, root), "blend", "exec"), env)
    out = pd.read_csv(root / "working/submission.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    return out.loc[ids, TARGETS], json.loads((root / "working/blend_receipt.json").read_text())


def test_noisyor_matches_the_verifier_formulas(tmp_path):
    ids = [f"1.2.{i}" for i in range(300)]
    parts = correlated(300, 1)
    out, receipt = run_noisyor(tmp_path, ids, parts)
    assert np.abs(out.to_numpy() - reference(parts, W)).max() < 1e-6
    assert receipt["method"] == "noisyor-sibling" and receipt["left_out"] == {} and receipt["every_part_failed"] == 0
    assert receipt["failed_studies"] == {"theirs": 0, "ours": 0, "r7": 0} and receipt["weights_used"] == W
    # the correction is applied to its three findings only
    plain = reference(parts, W, siblings={})
    changed = [label for j, label in enumerate(TARGETS) if np.abs(out[label].to_numpy() - plain[:, j]).max() > 1e-6]
    assert changed == ["Lateral OA", "PF OA", "Synovitis"]


def test_noisyor_differs_from_the_rank_blend_but_agrees_on_order_mostly(tmp_path):
    ids = [f"1.2.{i}" for i in range(300)]
    parts = correlated(300, 2)
    new, _ = run_noisyor(tmp_path / "a", ids, parts)
    old, _ = run_noisyor(tmp_path / "b", ids, parts, code=builder.BLEND)
    assert not np.allclose(new.to_numpy(), old.to_numpy())
    for label in TARGETS:
        assert new[label].corr(old[label], method="spearman") > 0.95


def test_noisyor_studies_a_part_failed_on_take_the_other_parts(tmp_path):
    ids = [f"1.2.{i}" for i in range(200)]
    parts = correlated(200, 3)
    parts["ours"][5] = np.nan  # an invalid row
    parts["theirs"][7] = 0.5  # a placeholder row: all 12 values equal
    failed = {name: np.zeros(200, bool) for name in parts}
    failed["ours"][[5, 10, 11, 12]] = True  # 10-12: our bundle gave these a stand-in prediction and listed them
    failed["theirs"][[7, 20]] = True  # 20: no series read by their sets
    failed["r7"][[30]] = True  # not scored by R7's workers
    failed["ours"][40] = failed["theirs"][40] = failed["r7"][40] = True  # every part failed: the median
    scored = [u for i, u in enumerate(ids) if i not in (30, 40)]
    out, receipt = run_noisyor(tmp_path, ids, parts, r7_scored=scored, ours_filled=[ids[10], ids[11], ids[12], ids[40]],
                               theirs_empty=[ids[20], ids[40], "1.2.not-a-test-study"])
    clean = {k: np.nan_to_num(v, nan=0.5) for k, v in parts.items()}
    assert np.abs(out.to_numpy() - reference(clean, W, failed)).max() < 1e-6
    assert receipt["failed_studies"] == {"theirs": 3, "ours": 5, "r7": 2} and receipt["every_part_failed"] == 1
    assert receipt["r7_not_scored"] == 2 and receipt["left_out"] == {} and receipt["flag_notes"] == {}
    assert receipt["flag_files"] == {"theirs": "read: 2 listed", "ours": "read: 4 listed"}


def test_noisyor_without_flag_files_or_failures_is_the_plain_method(tmp_path):
    ids = [f"1.2.{i}" for i in range(100)]
    parts = correlated(100, 4)
    a, _ = run_noisyor(tmp_path / "a", ids, parts)
    b, receipt = run_noisyor(tmp_path / "b", ids, parts, ours_filled=[], theirs_empty=[])
    assert a.equals(b) and receipt["failed_studies"] == {"theirs": 0, "ours": 0, "r7": 0}
    assert receipt["flag_files"] == {"theirs": "read: 0 listed", "ours": "read: 0 listed"}
    _, receipt = run_noisyor(tmp_path / "c", ids, parts)
    assert receipt["flag_files"] == {"theirs": "missing", "ours": "missing"}  # the check run must not pass on this


def test_noisyor_parts_left_out(tmp_path):
    ids = [f"1.2.{i}" for i in range(100)]
    parts = correlated(100, 5)
    for share in (3, 4, 70):  # no 3% rule here: R7 counts where it scored; other studies take the other parts
        out, receipt = run_noisyor(tmp_path / f"a{share}", ids, parts, r7_scored=ids[:100 - share])
        failed = {"r7": np.arange(100) >= 100 - share}
        assert receipt["left_out"] == {} and receipt["failed_studies"]["r7"] == share and "r7" in receipt["weights_used"]
        assert np.abs(out.to_numpy() - reference(parts, W, failed)).max() < 1e-6
    out, receipt = run_noisyor(tmp_path / "b", ids, parts, r7_scored=[])  # R7 scored nothing: left out
    assert receipt["left_out"] == {"r7": "left out: failed on every study"}
    assert np.abs(out.to_numpy() - reference({k: parts[k] for k in ("theirs", "ours")}, W)).max() < 1e-6
    out, receipt = run_noisyor(tmp_path / "c", ids, parts, ok={"theirs": True, "ours": False, "r7": True})
    assert receipt["left_out"] == {"ours": "stage failed"}
    assert np.abs(out.to_numpy() - reference({k: parts[k] for k in ("theirs", "r7")}, W)).max() < 1e-6
    out, receipt = run_noisyor(tmp_path / "d", ids, parts, ours_filled=ids)  # filled every study: left out
    assert receipt["left_out"] == {"ours": "left out: failed on every study"}
    bad = dict(parts, theirs=np.full((100, 12), np.nan))  # an unusable file is a failure on every study
    out, receipt = run_noisyor(tmp_path / "e", ids, bad)
    assert receipt["left_out"] == {"theirs": "left out: failed on every study"}
    with pytest.raises(RuntimeError, match="no stage produced"):
        run_noisyor(tmp_path / "f", ids, parts, ok={"theirs": False, "ours": False, "r7": False})


def test_noisyor_unreadable_flag_file_keeps_the_part(tmp_path):
    ids = [f"1.2.{i}" for i in range(50)]
    parts = correlated(50, 6)
    (tmp_path / "working").mkdir(parents=True)
    (tmp_path / "working/inference-summary.json").write_text("{not json")
    out, receipt = run_noisyor(tmp_path, ids, parts)
    assert "ours" in receipt["weights_used"] and receipt["flag_notes"]["ours"].startswith("unreadable")
    assert receipt["flag_files"]["ours"] == "unreadable"
    assert np.abs(out.to_numpy() - reference(parts, W)).max() < 1e-6


def test_noisyor_three_studies_and_ties(tmp_path):
    ids = ["1.2.0", "1.2.1", "1.2.2"]
    parts = correlated(3, 7)
    parts["r7"][:, 0] = 1.0  # R7 saturates: ties
    out, receipt = run_noisyor(tmp_path, ids, parts)
    values = out.to_numpy()
    assert values.shape == (3, 12) and ((values > 0) & (values <= 1)).all() and receipt["rows"] == 3
    assert np.abs(values - reference(parts, W)).max() < 1e-6


DUMMY_MASK = '''
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
out = Path(os.environ["TEAM_OUTPUT_DIR"]) / "six_model_predictions"
ids = pd.read_csv(sys.argv[0].replace("working/theirs/team_sets.py", "data/sample_submission.csv"), dtype=str)["StudyInstanceUID"]
cache = Path(sys.argv[0]).parents[2] / "temp/rsna_six_model_shared384_v1/0123456789abcdef0123"
cache.mkdir(parents=True, exist_ok=True)
mask = np.ones((len(ids), 5), dtype=np.float32)
mask[3] = 0
mask[5, :4] = 0  # one slot left: not a failure
np.save(cache / "test_mask.npy", mask)
np.save(cache / "test_study_ids.npy", ids.to_numpy())
for key in ("dino", "eff", "dino3"):
    d = out / key
    d.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(np.full((len(ids), 12), 0.3), columns=__TARGETS__)
    frame.insert(0, "StudyInstanceUID", ids)
    frame.to_csv(d / "test_probabilities.csv", index=False)
    (d / "complete.json").write_text("{}")
'''


def test_theirs_stage_records_studies_without_series(setup):
    root, ids, _ = setup
    stage = builder.replace(builder.THEIRS_STAGE, "finally:\n    shutil.rmtree(THEIRS_CACHE, ignore_errors=True)"
                            "  # about 23 GB; ours and R7 need the disk\n", builder.THEIRS_NO_SERIES, 1, "clean-up")
    code = stage.replace("__SETS__", repr(builder.SETS)).replace("__LIMIT__", "60").replace(
        "__SOURCE__", repr(DUMMY_MASK.replace("__TARGETS__", repr(TARGETS))))
    env = {"NB_T0": time.time()}
    exec(compile(localise(code, root), "theirs", "exec"), env)
    assert env["THEIRS_OK"]
    assert json.loads((root / "working/theirs/no_series_studies.json").read_text()) == [ids[3]]
    assert not (root / "temp/rsna_six_model_shared384_v1").exists(), "their image store must be deleted"


def test_built_notebook_postproc_changes_only_stage_4_and_the_record(tmp_path):
    if not Path("notebooks/rsna-ensemble.ipynb").is_file() or not Path(
            "artifacts/kaggle-oaiseeds6/notebook/inference.ipynb").is_file():
        pytest.skip("source notebooks not present")
    built = {}
    for name, extra in (("default", []), ("none", ["--postproc", "none"]), ("on", ["--postproc", "noisyor-sibling"])):
        sys.argv = ["build", "--output-dir", str(tmp_path / name), *extra]
        builder.main()
        built[name] = json.loads((tmp_path / name / "inference.ipynb").read_text())
    assert built["default"] == built["none"]
    same = [a == b for a, b in zip(built["default"]["cells"], built["on"]["cells"], strict=True)]
    assert same == [False, True, False, True, True, False]
    on = ["".join(c["source"]) for c in built["on"]["cells"]]
    assert on[5] == builder.BLEND_NOISYOR and "no_series_studies.json" in on[2] and "noisy-OR" in on[0]
    for source in on[1:]:
        compile(source, "cell", "exec")


def test_builder_rejects_a_zero_weight(tmp_path):
    if not Path("notebooks/rsna-ensemble.ipynb").is_file() or not Path(
            "artifacts/kaggle-oaiseeds6/notebook/inference.ipynb").is_file():
        pytest.skip("source notebooks not present")
    for extra in ([], ["--postproc", "noisyor-sibling"]):
        sys.argv = ["build", "--output-dir", str(tmp_path), "--blend-weights", "0.6,0.4,0", *extra]
        with pytest.raises(ValueError, match="above 0"):
            builder.main()
