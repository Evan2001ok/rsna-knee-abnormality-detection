#!/usr/bin/env python3
"""Build the merged team's combined Kaggle notebook (Oct 15, 2026).

A "set" is one of the teammates' models with its folds (eff, dino, dino3, conv, newconv). "Theirs", "ours" and "R7"
(heliosli's public ConvNeXt-small) are the three parts of the blend.

Stage 1, theirs: some of the teammates' model sets from their 0.965 notebook (rsna-ensemble version 1, saved as
  notebooks/rsna-ensemble.ipynb): eff, dino and dino3 by default; the final notebook also ran newconv (--their-sets
  dino,eff,dino3,newconv). They run unchanged except for which sets run, where files go
  and that a smaller set count is accepted. It runs as a separate process with a hard time limit; a set that finishes
  writes its own file, so a cut-off keeps the finished sets. Their rule: equal mean of the sets' probabilities.
Stage 2, ours: our bundle cell, exactly as in our earlier "six-seed" notebook (six of our seeds blended with R7, in the cell layout scripts/build_s6_blend_notebook.py --model r7 writes), with whichever
  bundle is attached (56 prediction windows).
Stage 3, R7: heliosli's setup and run cells, exactly as in that notebook (its own stop at 8.3 h from the notebook start).
Stage 4: per-finding percentile-rank blend theirs 0.5, ours 0.3, R7 0.2 (tested out of fold, rules fixed beforehand).
  A part that failed is left out and the weights of the rest are rescaled; R7 is left out if it scored under 97% of
  the studies, and a study R7 did not score takes the other parts' blend.
Stages 2 and 3 run inside try/except, so one failing part cannot stop the notebook.

--postproc noisyor-sibling (Oct 17, 2026; default none keeps the notebook byte-identical): stage 4 blends the ranks as
a noisy-OR and applies a fixed correction between related findings, and a study one part failed on is ranked without
that part and takes the other parts' score (the "insurance").

--hard-limits (Oct 20, 2026; default off keeps the notebook byte-identical): three runs of this notebook passed
Kaggle's 9 h because a call in stage 2 or 3 never returned, and only stage 1 had a hard stop. With the flag, stage 2 runs as its own process (its image-preparation workers start clean
instead of being copied from a running process) and is stopped at --ours-hard-hours on the notebook clock, keeping the
64-study groups it finished; R7's controller is stopped at --r7-hard-hours and its submission rebuilt from the rows its
workers finished, as the controller itself would; the remaining waits get time limits. A normal run gives the same files.

--timing builds instead a script kernel that runs only stage 1 on 64 training studies laid out as a fake test set,
to measure their sets' time on Kaggle's two T4s.

Run: .venv/bin/python scripts/build_team_combined_notebook.py --output-dir artifacts/kaggle-team3/notebook
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COMPETITION_PATH = "Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')"
SETS = ("dino", "eff", "dino3")  # run order: if the time limit cuts stage 1, the cheaper sets have finished
# The teammates' newconv weights are in their private Kaggle dataset, so its name is a placeholder here. To build a
# notebook with newconv (--their-sets ...,newconv), put the "owner/name" of a Kaggle dataset holding those weights.
# Their public weights dataset (lihu6666/rsna-knee-model-10th-multi-seed-team) has this run's checkpoints one folder
# deeper than _link_newconv below looks; we have not tried building from it.
NEWCONV_DATASET = "<owner>/<newconv-dataset>"
THEIR_DATASETS = {"eff": "minminandy/distillstage1effs384", "dino": "zhuyifanss/distillstage1dinov2s384",
                  "conv": "minminandy/distillstage1conv384",
                  "dino3": "lihu6666/dinov3-convnext-tiny-lihu-of-fork-of-rsna-cd4e8a",
                  "newconv": NEWCONV_DATASET}
# The teammate's newconv inference notebook, saved locally (not part of this repository).
NEWCONV_NOTEBOOK = Path("artifacts/teammates/newconv-20261016/newconv-lihu-fork-of-cd4e8a.ipynb")

# Oct 17, 2026: newconv, a teammate's 0.965 ConvNeXt-tiny model with its rotation views, as one more set of their
# ensemble. Their shared scan preparation is identical to newconv's own (same functions and settings), so newconv reads
# the shared cache; its four checkpoints sit in two folders of the teammate's dataset and are linked into one folder
# here; only newconv keeps three views.
NEWCONV_SETUP = """# Combined team notebook (Oct 17, 2026): a teammate's 0.965 model (newconv) as a seventh set.
_NEWCONV_RUN = 'rsna_convnext_tiny_384_oai8649_mix58_42_seven_graycopy24'
_NEWCONV_ROOT = Path('/kaggle/temp/newconv_run')
MODEL_SPECS.append({'run_root': str(_NEWCONV_ROOT),
                    'dataset_url': 'https://www.kaggle.com/datasets/__NEWCONV_DATASET__',
                    'model_name': 'convnext_tiny.fb_in22k_ft_in1k_384', 'feature_dim': 768, 'model_input_size': 384,
                    'tokens_per_slot': 24, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'key': 'newconv',
                    'dataset_slug': '__NEWCONV_SLUG__', 'run_name': _NEWCONV_RUN})
MODEL_TTA_ANGLES['newconv'] = (-12.0, 0.0, 12.0)


def _link_newconv():
    mounts = [Path('/kaggle/input/datasets/__NEWCONV_DATASET__'),
              Path('/kaggle/input/__NEWCONV_SLUG__')]
    for fold in range(4):
        found = sorted({p.resolve() for m in mounts if m.is_dir()
                        for p in m.glob(f'*/{_NEWCONV_RUN}/fold{fold}/fold0_final.pt')})
        if len(found) != 1:
            raise FileNotFoundError(f'newconv fold{fold}: expected one fold0_final.pt, found {found}')
        target = _NEWCONV_ROOT / f'fold{fold}' / 'fold0_final.pt'
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(found[0])
        print('newconv fold', fold, '->', found[0], flush=True)


if 'newconv' in _KEEP:
    _link_newconv()

""".replace("__NEWCONV_DATASET__", NEWCONV_DATASET).replace("__NEWCONV_SLUG__", NEWCONV_DATASET.split("/")[1])
NEWCONV_HOOK = """    if spec['key'] == 'newconv':  # Oct 17: the teammate's checkpoints (linked folder) and backbone chunk size
        backend.FOLD_CHECKPOINTS = {f: root / f'fold{f}' / 'fold0_final.pt' for f in backend.FOLDS}
        backend.BACKBONE_CHUNK_SIZE = 72
"""
R7_DATASETS = ["heliosli/rsna-knee-compact-r7cnxfsss", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"]
DOCKER = "gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"

CREDIT = """# Knee: merged team ensemble (theirs + ours + R7)

Stage 1 runs three of the team's six ConvNeXt / EfficientNet / DINO sets (the team's 0.965 notebook, unchanged apart
from the set list and file locations). Stage 2 runs our CoAtNet-2 bundle. Stage 3 runs heliosli's public ConvNeXt-small
"R7" as published in [heliosli's notebook](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click),
with the public datasets
[heliosli/rsna-knee-compact-r7cnxfsss](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss) and
[heliosli/rsna-orthofoundation-fold0-epoch7-20261004](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004).
Stage 4 averages per-finding percentile ranks with fixed weights (theirs 0.5, ours 0.3, R7 0.2).
"""

START = """# Notebook clock (every stage's time limit counts from here).
import time
NB_T0 = time.time()
"""

THEIRS_STAGE = """# Stage 1: the team's sets, as a separate process with a hard time limit; finished sets are kept if it is cut off.
import csv, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
THEIRS_SETS = __SETS__
THEIRS_LIMIT_SECONDS = __LIMIT__
THEIRS_DIR = Path('/kaggle/working/theirs')
THEIRS_CACHE = Path('/kaggle/temp/rsna_six_model_shared384_v1')
THEIRS_DIR.mkdir(parents=True, exist_ok=True)
(THEIRS_DIR / 'team_sets.py').write_text(__SOURCE__, encoding='utf-8')
THEIRS_OK, THEIRS_DONE, THEIRS_RC = False, [], None
try:
    with open(THEIRS_DIR / 'log.txt', 'w') as log:
        proc = subprocess.Popen([sys.executable, '-u', str(THEIRS_DIR / 'team_sets.py')], stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(THEIRS_DIR), start_new_session=True,
                                env=dict(os.environ, TEAM_OUTPUT_DIR=str(THEIRS_DIR)))
        try:
            THEIRS_RC = proc.wait(timeout=THEIRS_LIMIT_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            THEIRS_RC = 'time limit'
    print(''.join((THEIRS_DIR / 'log.txt').read_text(encoding='utf-8', errors='replace').splitlines(True)[-25:]), flush=True)
    for key in THEIRS_SETS:
        folder = THEIRS_DIR / 'six_model_predictions' / key
        if (folder / 'complete.json').is_file() and (folder / 'test_probabilities.csv').is_file():
            THEIRS_DONE.append(key)
    if THEIRS_DONE:
        frames = []
        for key in THEIRS_DONE:
            with open(THEIRS_DIR / 'six_model_predictions' / key / 'test_probabilities.csv', newline='') as handle:
                reader = csv.DictReader(handle)
                frames.append((reader.fieldnames, {row['StudyInstanceUID']: row for row in reader}))
        columns = frames[0][0]
        ids = list(frames[0][1])
        for cols, rows in frames:
            assert cols == columns and sorted(rows) == sorted(ids), 'their sets disagree on studies or columns'
        with open(Path('/kaggle/working/theirs_sub.csv'), 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for u in ids:
                values = [sum(float(rows[u][c]) for _, rows in frames) / len(frames) for c in columns[1:]]
                assert all(0.0 <= v <= 1.0 for v in values)
                writer.writerow([u] + ['%.8f' % v for v in values])
        THEIRS_OK = True
except Exception as error:
    print('THEIRS_STAGE_FAILED', repr(error)[:1000], flush=True)
finally:
    shutil.rmtree(THEIRS_CACHE, ignore_errors=True)  # about 23 GB; ours and R7 need the disk
    for stray in (Path('/kaggle/working/submission.csv'),):
        if stray.exists():
            stray.unlink()
THEIRS_SECONDS = time.time() - NB_T0
print('THEIRS_STAGE_DONE', json.dumps({'ok': THEIRS_OK, 'sets': THEIRS_DONE, 'exit': THEIRS_RC,
                                       'seconds': round(THEIRS_SECONDS)}), flush=True)
"""

OURS_STAGE = """# Stage 2: our bundle (cell unchanged from the six-seed notebook), inside try/except.
OURS_OK = False
try:
    exec(compile(__OURS__, 'ours_stage', 'exec'), globals())
    import csv, os
    from pathlib import Path
    _work = Path('/kaggle/working')
    assert (_work / 'submission.csv').is_file(), 'our stage wrote no submission.csv'
    os.replace(_work / 'submission.csv', _work / 'ours_sub.csv')
    with open(_work / 'ours_sub.csv', newline='') as handle:
        _rows = list(csv.DictReader(handle))
    if all(float(v) == 0.5 for r in _rows for k, v in r.items() if k != 'StudyInstanceUID'):
        raise RuntimeError('our stage produced only the 0.5 fallback')
    OURS_OK = True
except Exception as error:
    print('OURS_STAGE_FAILED', repr(error)[:1000], flush=True)
OURS_SECONDS = time.time() - NB_T0
print('OURS_STAGE_DONE', json.dumps({'ok': OURS_OK, 'seconds': round(OURS_SECONDS)}), flush=True)
"""

R7_STAGE = """# Stage 3: heliosli's R7 (setup and run cells unchanged from the six-seed notebook), inside try/except.
R7_OK = False
try:
    exec(compile(__SETUP__, 'r7_setup', 'exec'), globals())
__WARM_START__    exec(compile(__RUN__, 'r7_run', 'exec'), globals())
    R7_OK = True
except Exception as error:
    print('R7_STAGE_FAILED', repr(error)[:1000], flush=True)
__WARM_STOP__print('R7_STAGE_DONE', json.dumps({'ok': R7_OK, 'seconds': round(time.time() - NB_T0)}), flush=True)
"""

# Oct 16, 2026 (our speed tests): R7's workers read their DICOMs one file at a time from
# /kaggle/input (about 58 MB/s; 474 MB/s with 16 threads). This reader thread follows each worker's own study order
# (worker*/ids.json, written by R7's controller) and its progress (rows in worker*/predictions.csv) and reads the next
# few studies' files ahead with many threads, so R7 finds them in memory. R7's code and outputs are unchanged.
R7_WARM_START = """    import concurrent.futures as _cf, threading as _threading
    _R7_WARM_STOP = _threading.Event()
    _R7_WARM_COUNT = [0]

    def _r7_warm(data, pred, ahead=8, threads=12):
        pool, done, orders = _cf.ThreadPoolExecutor(threads), set(), {}

        def read_study(uid):
            for root, _, files in os.walk(data / 'test_series' / uid):
                for name in files:
                    try:
                        with open(os.path.join(root, name), 'rb') as handle:
                            while handle.read(8 << 20):
                                pass
                    except OSError:
                        pass

        while not _R7_WARM_STOP.is_set():
            for folder in sorted(pred.glob('worker*')):
                if folder.name not in orders and (folder / 'ids.json').is_file():
                    try:
                        orders[folder.name] = json.loads((folder / 'ids.json').read_text())
                    except ValueError:
                        continue
            for name, ids in orders.items():
                rows = pred / name / 'predictions.csv'
                try:
                    finished = max(0, sum(1 for _ in open(rows)) - 1) if rows.is_file() else 0
                except OSError:
                    finished = 0
                for uid in ids[finished:finished + ahead]:
                    if uid not in done:
                        done.add(uid)
                        _R7_WARM_COUNT[0] += 1
                        pool.submit(read_study, uid)
            _R7_WARM_STOP.wait(2)
        pool.shutdown(wait=False, cancel_futures=True)

    _R7_WARMER = _threading.Thread(target=_r7_warm, args=(DATA, TMP / 'prediction'), daemon=True)
    _R7_WARMER.start()
"""
R7_WARM_STOP = """if '_R7_WARM_STOP' in globals():
    _R7_WARM_STOP.set()
    print('R7_WARM', json.dumps({'studies_read_ahead': _R7_WARM_COUNT[0]}), flush=True)
"""

# Oct 16, 2026 (our speed tests): one long-lived prediction process per GPU, the next group prepared while the current
# one predicts, autocast + channels_last + cudnn.benchmark + compiled encoders, 64-study groups. Within 2e-4 of the
# submitted mode on 400 studies (AUC +0.0000 on all three label sets we score with). A bundle whose code lacks these
# options runs in the submitted mode.
OURS_FAST = """import inspect
OURS_MODE = {"persistent": True, "prepare_ahead": True, "accelerate": True, "studies_per_chunk": 64}
if not all(name in inspect.signature(run_bundle).parameters for name in OURS_MODE):
    OURS_MODE = {"separate_processes": True}
print(json.dumps({"event": "ours_mode", "mode": OURS_MODE}), flush=True)
"""

# --hard-limits (Oct 20, 2026). Stage 2 as its own process: the cell's
# code runs unchanged inside it. Kept under the main guard because spawned children import this file as __mp_main__.
OURS_SCRIPT = """# Our stage as its own process (written and started by the notebook's stage-2 cell; Oct 20, 2026).
import json, multiprocessing, os, sys, time
OURS_SOURCE = __OURS__
if __name__ == '__main__':
    # Image preparation (raptor_data's process pool) starts clean worker processes instead of copying this one.
    multiprocessing.set_start_method('spawn', force=True)
    NB_T0 = float(os.environ['NB_T0'])
    exec(compile(OURS_SOURCE, 'ours_stage', 'exec'), globals())
"""

OURS_STAGE_HARD = """# Stage 2: our bundle (cell unchanged from the six-seed notebook) as a separate process with a hard time limit (Oct 20,
# 2026). If it is still running at the limit it is stopped and the 64-study groups it finished are kept; a study it did
# not reach takes the other parts' score in the blend.
OURS_OK, OURS_RC, OURS_SALVAGED = False, None, None
OURS_HARD_SECONDS = __OURS_HARD__
try:
    import csv, io, math, os, signal, subprocess, sys
    from pathlib import Path
    _work = Path('/kaggle/working')
    _ours_dir = Path('/kaggle/temp/ours_stage')
    _ours_dir.mkdir(parents=True, exist_ok=True)
    (_ours_dir / 'ours_stage.py').write_text(__OURS_SCRIPT__, encoding='utf-8')
    with open(_ours_dir / 'log.txt', 'w') as _log:
        _proc = subprocess.Popen([sys.executable, '-u', str(_ours_dir / 'ours_stage.py')], stdout=_log,
                                 stderr=subprocess.STDOUT, cwd=str(_work), start_new_session=True,
                                 env=dict(os.environ, NB_T0=repr(NB_T0)))
        try:
            OURS_RC = _proc.wait(timeout=max(60, NB_T0 + OURS_HARD_SECONDS - time.time()))
            try:
                os.killpg(_proc.pid, signal.SIGKILL)  # anything the stage left running if it died abnormally (normally nothing)
            except (ProcessLookupError, PermissionError):
                pass
        except subprocess.TimeoutExpired:
            os.killpg(_proc.pid, signal.SIGKILL)
            try:
                _proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                pass
            OURS_RC = 'time limit'
    print((_ours_dir / 'log.txt').read_text(encoding='utf-8', errors='replace'), flush=True)
    if not ((_work / 'submission.csv').is_file() and (_work / 'inference-summary.json').is_file()):
        # run_bundle did not finish: rebuild its file from the groups it finished, the rest filled with their mean (as
        # run_bundle fills studies it did not reach) and listed as filled, so the blend gives them the other parts' score.
        import shutil
        for _folder in _work.glob('knee-images-*'):  # half-prepared image caches (about 1 GB per group) left by the stop
            shutil.rmtree(_folder, ignore_errors=True)
        _roots = [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), Path('/kaggle/input/rsna-knee-abnormality-detection')]
        _data = next(r for r in _roots if (r / 'sample_submission.csv').is_file())
        with open(_data / 'sample_submission.csv', newline='') as handle:
            _reader = csv.reader(handle)
            _header = next(_reader)
            _ids = [row[0] for row in _reader]
        _labels, _rows = _header[1:], {}
        for _path in sorted((_work / 'prediction-parts').glob('predictions-*.csv')):
            if '-device' in _path.name:
                continue
            _text = _path.read_text(encoding='utf-8', errors='replace')
            for _row in csv.DictReader(io.StringIO(_text[:_text.rfind('\\n') + 1])):
                try:
                    _values = [float(_row[label]) for label in _labels]
                except (KeyError, TypeError, ValueError):
                    continue
                if all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in _values):
                    _rows[_row['StudyInstanceUID']] = _values
        _got = [u for u in _ids if u in _rows]
        OURS_SALVAGED = len(_got)
        if _got:
            _mean = [sum(_rows[u][j] for u in _got) / len(_got) for j in range(len(_labels))]
            _missing = [u for u in _ids if u not in _rows]
            with open(_work / 'submission.csv', 'w', newline='') as handle:
                _writer = csv.writer(handle)
                _writer.writerow(_header)
                for u in _ids:
                    _writer.writerow([u] + [repr(v) for v in _rows.get(u, _mean)])
            (_work / 'inference-summary.json').write_text(json.dumps({'hard_stop': OURS_RC, 'salvaged_studies': len(_got),
                                                                       'filled_study_ids': _missing}))
    assert (_work / 'submission.csv').is_file(), 'our stage wrote no submission.csv'
    os.replace(_work / 'submission.csv', _work / 'ours_sub.csv')
    with open(_work / 'ours_sub.csv', newline='') as handle:
        _rows = list(csv.DictReader(handle))
    if all(float(v) == 0.5 for r in _rows for k, v in r.items() if k != 'StudyInstanceUID'):
        raise RuntimeError('our stage produced only the 0.5 fallback')
    OURS_OK = True
except Exception as error:
    print('OURS_STAGE_FAILED', repr(error)[:1000], flush=True)
OURS_SECONDS = time.time() - NB_T0
print('OURS_STAGE_DONE', json.dumps({'ok': OURS_OK, 'exit': OURS_RC, 'salvaged': OURS_SALVAGED,
                                     'seconds': round(OURS_SECONDS)}), flush=True)
"""

# --early-decoders (Oct 20, 2026): the host's data page lists JPEG
# Lossless among the test files' formats, which the image's Pillow cannot decode, so the teammates' reader would drop such a
# series. Only the four decoder plugins are installed, from our bundle's own wheels, with no dependencies (the image already
# has pydicom 3.0.2 and NumPy), before stage 1 and with a time limit; if anything fails, stage 1 runs as before.
EARLY_DECODERS = """# Before stage 1: the decoder plugins for compressed DICOMs (JPEG Lossless, JPEG-LS), from our bundle's wheels (Oct 20, 2026).
import json, os, signal, subprocess, sys, time, zipfile
from pathlib import Path
EARLY_DECODERS = {'installed': False}
try:
    _roots = [p for p in Path('/kaggle/input').iterdir() if p.is_dir() and p.name not in ('competitions', 'rsna-knee-abnormality-detection')]
    _wheel_dir = next((f.parent / 'wheelhouse' for r in _roots for f in r.glob('**/bundle.json') if (f.parent / 'wheelhouse').is_dir()), None)
    if _wheel_dir is None:  # Kaggle may keep the uploaded archive as one zip file
        _wheel_dir = Path('/kaggle/temp/early_wheels')
        for _zip in (z for r in _roots for z in r.glob('**/*.zip')):
            with zipfile.ZipFile(_zip) as _archive:
                _names = _archive.namelist()
                if any(Path(n).name == 'bundle.json' for n in _names):
                    _wheel_dir.mkdir(parents=True, exist_ok=True)
                    for _n in _names:
                        if _n.endswith('.whl') and Path(_n).parent.name == 'wheelhouse':
                            (_wheel_dir / Path(_n).name).write_bytes(_archive.read(_n))
                    break
    _want = ('pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg', 'pyjpegls')
    _wheels = sorted(str(p) for p in _wheel_dir.glob('*.whl') if p.name.split('-')[0].lower().replace('_', '-') in _want)
    EARLY_DECODERS['wheels'] = [Path(w).name for w in _wheels]
    if len(_wheels) == len(_want):
        _proc = subprocess.Popen([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *_wheels], start_new_session=True)
        try:
            EARLY_DECODERS['installed'] = _proc.wait(timeout=300) == 0
        except subprocess.TimeoutExpired:
            os.killpg(_proc.pid, signal.SIGKILL)
            try:
                _proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    _check = subprocess.run([sys.executable, '-c', "import json, pydicom; from pydicom.pixels.decoders import "
                             "JPEGLosslessSV1Decoder as a, JPEGLSLosslessDecoder as b, JPEG2000LosslessDecoder as c; "
                             "print(json.dumps({'pydicom': pydicom.__version__, 'jpeg_lossless_sv1': a.is_available, "
                             "'jpeg_ls_lossless': b.is_available, 'jpeg2000_lossless': c.is_available}))"],
                            capture_output=True, text=True, timeout=60)
    EARLY_DECODERS['available'] = json.loads(_check.stdout.strip().splitlines()[-1]) if _check.returncode == 0 else _check.stderr[-300:]
except Exception as _error:
    EARLY_DECODERS['error'] = repr(_error)[:300]
print('EARLY_DECODERS', json.dumps(EARLY_DECODERS), flush=True)

"""

# R7's controller in its own process group, stopped at the hard limit; then its submission is rebuilt exactly as its
# controller builds it (probe_dual.py: each worker's finished rows, the rest filled with the column mean).
R7_HARD_HELPER = """import threading
R7_HARD_SECONDS, R7_HARD_STOP, R7_SETUP_SECONDS = __R7_HARD__, False, 1800


def _bounded_run(cmd, seconds, check=False):
    # subprocess.run's own timeout waits without a limit after its kill; this gives up 10 s after the kill
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        rc = proc.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        print('R7_SETUP_COMMAND_TIMEOUT', ' '.join(map(str, cmd[:3])), flush=True)
        if check:
            raise RuntimeError('R7 setup command timed out: ' + ' '.join(map(str, cmd[:4])))
        return None
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc


def _r7_setup_bounded(code, seconds):
    # R7's setup (copies, unzip, hashing, pip) in a helper thread, given up after `seconds`: a stalled read then
    # costs R7's part, not the whole run
    box = {}

    def target():
        try:
            exec(code, globals())
        except BaseException as error:
            box['error'] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise RuntimeError('R7 setup still running after %d s; R7 left out' % seconds)
    if 'error' in box:
        raise box['error']


def _r7_controller(cmd, env):
    global R7_HARD_STOP
    import csv, io
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        rc = proc.wait(timeout=max(60, NB_T0 + R7_HARD_SECONDS - time.time()))
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # anything the controller left running if it died abnormally (normally nothing)
        except (ProcessLookupError, PermissionError):
            pass
        return rc
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        try:
            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            pass
    R7_HARD_STOP = True
    out, data = Path(cmd[cmd.index('--output') + 1]), Path(cmd[cmd.index('--data') + 1])
    with open(data / 'sample_submission.csv', encoding='utf-8') as fh:
        rd = csv.reader(fh)
        header = next(rd)
        sample_ids = [r[0] for r in rd]
    labels, pred_rows = header[1:], {}
    for f in sorted(out.glob('worker*/predictions.csv')):
        text = f.read_text(encoding='utf-8', errors='replace')
        if not text.endswith('\\n'):  # a row cut off by the stop: drop it here and in the file the blend reads
            text = text[:text.rfind('\\n') + 1]
            f.write_text(text, encoding='utf-8')
        for r in csv.DictReader(io.StringIO(text)):
            try:
                v = [float(r[L]) for L in labels]
            except (KeyError, ValueError, TypeError):
                continue
            if all(0. <= x <= 1. for x in v):
                pred_rows[r['StudyInstanceUID']] = v
    scored = [u for u in sample_ids if u in pred_rows]
    mean = [sum(pred_rows[u][j] for u in scored) / len(scored) for j in range(len(labels))] if scored else [0.5] * len(labels)
    with open(out / 'submission.csv', 'w', encoding='utf-8', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        for u in sample_ids:
            wr.writerow([u] + ['%.9g' % x for x in pred_rows.get(u, mean)])
    receipt = {'status': 'HARD_STOP' if scored else 'ALL_FALLBACK', 'rows': len(sample_ids), 'scored': len(scored),
               'fallback_studies': len(sample_ids) - len(scored), 'seconds': round(time.time() - NB_T0, 1)}
    (out / 'submission_receipt.json').write_text(json.dumps(receipt, indent=1), encoding='utf-8')
    print('R7_HARD_STOP', json.dumps(receipt), flush=True)
    return 0


"""

BLEND = """# Stage 4: per-finding rank blend, weights fixed before any leaderboard result: theirs 0.5, ours 0.3, R7 0.2.
# A failed part is left out and the others rescaled; R7 is left out under 97% scored studies; a study R7 did not score
# takes the other parts' blend.
import csv, json, math, time
from pathlib import Path
WEIGHTS = {'theirs': 0.5, 'ours': 0.3, 'r7': 0.2}
work = Path('/kaggle/working')
_roots = [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), Path('/kaggle/input/rsna-knee-abnormality-detection')]
_data = next(r for r in _roots if (r / 'sample_submission.csv').is_file())
with open(_data / 'sample_submission.csv', newline='') as handle:
    _reader = csv.DictReader(handle)
    columns = _reader.fieldnames
    test_ids = [row['StudyInstanceUID'] for row in _reader]
labels = columns[1:]
n = len(test_ids)


def read(path):
    with open(path, newline='') as handle:
        rows = {row['StudyInstanceUID']: row for row in csv.DictReader(handle)}
    assert sorted(rows) == sorted(test_ids), '%s: rows differ from the test set' % path
    out = {}
    for label in labels:
        values = [float(rows[u][label]) for u in test_ids]
        assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values), '%s: bad values' % path
        out[label] = values
    return out


parts, notes = {}, {}
for name, ok, path in (('theirs', globals().get('THEIRS_OK'), work / 'theirs_sub.csv'),
                       ('ours', globals().get('OURS_OK'), work / 'ours_sub.csv'),
                       ('r7', globals().get('R7_OK'), work / 'r7_sub.csv')):
    if not ok:
        notes[name] = 'stage failed'
        continue
    try:
        parts[name] = read(path)
    except Exception as error:
        notes[name] = 'unreadable: ' + repr(error)[:300]
r7_missing = [False] * n
if 'r7' in parts:
    scored = set()  # R7 fills a study it did not score with the column mean: read what its workers actually scored
    for path in pred.glob('worker*/predictions.csv'):
        with open(path, newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                try:
                    values = [float(row[label]) for label in labels]
                except (KeyError, ValueError, TypeError):
                    continue
                if all(0.0 <= v <= 1.0 for v in values):
                    scored.add(row['StudyInstanceUID'])
    r7_missing = [u not in scored for u in test_ids]
    if sum(r7_missing) > 0.03 * n:
        notes['r7'] = 'left out: %d of %d studies not scored' % (sum(r7_missing), n)
        del parts['r7']
        r7_missing = [False] * n
if not parts:
    raise RuntimeError('no stage produced predictions: %s' % notes)


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


blended = {}
for label in labels:
    ranks = {name: pct(part[label]) for name, part in parts.items()}
    column = []
    for i in range(n):
        use = [name for name in ranks if not (name == 'r7' and r7_missing[i])]
        total = sum(WEIGHTS[name] for name in use)
        column.append(sum(WEIGHTS[name] * ranks[name][i] for name in use) / total)
    blended[label] = column
with open(work / 'submission.csv', 'w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(columns)
    for i, u in enumerate(test_ids):
        values = [blended[label][i] for label in labels]
        assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values)
        writer.writerow([u] + ['%.6f' % v for v in values])
receipt = {'weights_used': {k: WEIGHTS[k] for k in parts}, 'left_out': notes, 'rows': n,
           'r7_not_scored': sum(r7_missing), 'theirs_sets': globals().get('THEIRS_DONE'),
           'theirs_seconds': round(globals().get('THEIRS_SECONDS', 0)), 'ours_end_seconds': round(globals().get('OURS_SECONDS', 0)),
           'total_seconds': round(time.time() - NB_T0)}
(work / 'blend_receipt.json').write_text(json.dumps(receipt, indent=1))
print('NOTEBOOK_COMPLETE', json.dumps(receipt), flush=True)
"""

# Oct 17, 2026 (option 2 of our post-processing search, plus the "insurance": a study one part failed on takes the other
# parts' score): with --postproc noisyor-sibling, the theirs stage records the studies its shared image cache read no
# series for (before it deletes the cache), and stage 4 is BLEND_NOISYOR.
THEIRS_NO_SERIES = """finally:
    try:  # Oct 17 insurance: studies for which their shared image cache read no series at all (all five slots empty)
        import numpy as _np
        _masks = sorted(THEIRS_CACHE.glob('*/test_mask.npy'))
        if len(_masks) == 1:
            _mask = _np.load(_masks[0])
            _ids = [str(u) for u in _np.load(_masks[0].parent / 'test_study_ids.npy', allow_pickle=True)]
            assert _mask.shape == (len(_ids), 5), 'their mask has an unexpected shape'
            _empty = [u for u, m in zip(_ids, _mask) if not m.any()]
            (THEIRS_DIR / 'no_series_studies.json').write_text(json.dumps(_empty))
            print('THEIRS_NO_SERIES', len(_empty), flush=True)
        else:
            print('THEIRS_NO_SERIES unknown: found %d masks' % len(_masks), flush=True)
    except Exception as error:
        print('THEIRS_NO_SERIES unknown:', repr(error)[:300], flush=True)
    shutil.rmtree(THEIRS_CACHE, ignore_errors=True)  # about 23 GB; ours and R7 need the disk
"""

BLEND_NOISYOR = """# Stage 4 (Oct 17, 2026): per finding, a noisy-OR of the parts'
# percentile ranks, score = sum of w * -log(1 - rank + 0.5/m) / sum of w, weights theirs 0.5, ours 0.3, R7 0.2; then, on
# normal scores of its ranks, Lateral OA - 0.10 x Medial OA, PF OA - 0.10 x Medial OA, Synovitis - 0.20 x Effusion.
# Insurance: a study a part failed on (row missing or invalid, all 12 values equal, filled in by the part itself, or no
# series read by their sets) is ranked without that part and takes the other parts' score. A failed stage, or a part that
# failed on every study, is left out and the others rescaled. No 3% rule for R7 (decided Oct 17): a study R7
# did not reach before its stop takes the other parts' score, and R7 still counts on every study it scored.
import csv, json, math, statistics, time
from pathlib import Path
WEIGHTS = {'theirs': 0.5, 'ours': 0.3, 'r7': 0.2}
SIBLINGS = {'Lateral OA': ('Medial OA', -0.10), 'PF OA': ('Medial OA', -0.10), 'Synovitis': ('Effusion', -0.20)}
work = Path('/kaggle/working')
_roots = [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), Path('/kaggle/input/rsna-knee-abnormality-detection')]
_data = next(r for r in _roots if (r / 'sample_submission.csv').is_file())
with open(_data / 'sample_submission.csv', newline='') as handle:
    _reader = csv.DictReader(handle)
    columns = _reader.fieldnames
    test_ids = [row['StudyInstanceUID'] for row in _reader]
labels = columns[1:]
n = len(test_ids)
assert len(set(test_ids)) == n and all(a in labels and b in labels for a, (b, _) in SIBLINGS.items())


def read(path):
    # a part's values per finding (None on a study it failed on) and the studies it failed on
    with open(path, newline='') as handle:
        reader = csv.DictReader(handle)
        assert set(labels) <= set(reader.fieldnames or []), '%s: columns differ' % path
        rows = {row['StudyInstanceUID']: row for row in reader}
    out, failed = {label: [None] * n for label in labels}, set()
    for i, u in enumerate(test_ids):
        try:
            values = [float(rows[u][label]) for label in labels]
        except (KeyError, TypeError, ValueError):
            failed.add(u)
            continue
        if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values) or len(set(values)) == 1:
            failed.add(u)
            continue
        for label, v in zip(labels, values):
            out[label][i] = v
    return out, failed


parts, failed, notes, flag_notes, flag_files = {}, {}, {}, {}, {}
for name, ok, path in (('theirs', globals().get('THEIRS_OK'), work / 'theirs_sub.csv'),
                       ('ours', globals().get('OURS_OK'), work / 'ours_sub.csv'),
                       ('r7', globals().get('R7_OK'), work / 'r7_sub.csv')):
    if not ok:
        notes[name] = 'stage failed'
        continue
    try:
        parts[name], failed[name] = read(path)
    except Exception as error:
        notes[name] = 'unreadable: ' + repr(error)[:300]
for name, path, key in (('theirs', work / 'theirs' / 'no_series_studies.json', None),
                        ('ours', work / 'inference-summary.json', 'filled_study_ids')):
    if name not in parts:
        continue
    flag_files[name] = 'missing'
    if path.is_file():  # studies the part filled in itself
        try:
            listed = json.loads(path.read_text())
            found = set(map(str, listed[key] if key else listed)) & set(test_ids)
            failed[name] |= found
            flag_files[name] = 'read: %d listed' % len(found)
        except Exception as error:
            flag_notes[name] = 'unreadable: ' + repr(error)[:300]
            flag_files[name] = 'unreadable'
r7_not_scored = 0
if 'r7' in parts:
    scored = set()  # R7 fills a study it did not score with the column mean: read what its workers actually scored
    for path in pred.glob('worker*/predictions.csv'):
        with open(path, newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                try:
                    values = [float(row[label]) for label in labels]
                except (KeyError, ValueError, TypeError):
                    continue
                if all(0.0 <= v <= 1.0 for v in values):
                    scored.add(row['StudyInstanceUID'])
    r7_not_scored = sum(u not in scored for u in test_ids)
    failed['r7'] |= {u for u in test_ids if u not in scored}
for name in list(parts):
    if len(failed[name]) >= n:
        notes[name] = 'left out: failed on every study'
        del parts[name]
if not parts:
    raise RuntimeError('no stage produced predictions: %s' % notes)


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


keep = {name: [i for i, u in enumerate(test_ids) if u not in failed[name]] for name in parts}
every_part_failed = [u for u in test_ids if all(u in failed[name] for name in parts)]
blended = {}
for label in labels:
    terms = {}
    for name, part in parts.items():
        c = 0.5 / len(keep[name])
        ranks = pct([part[label][i] for i in keep[name]])
        terms[name] = dict(zip(keep[name], (-math.log(1.0 - r + c) for r in ranks)))
    column = []
    for i in range(n):
        use = [name for name in terms if i in terms[name]]
        column.append(sum(WEIGHTS[name] * terms[name][i] for name in use) / sum(WEIGHTS[name] for name in use) if use else None)
    middle = statistics.median([v for v in column if v is not None])
    blended[label] = [middle if v is None else v for v in column]
_inv = statistics.NormalDist().inv_cdf
normal = {label: [_inv(r - 0.5 / n) for r in pct(blended[label])] for label in labels}
final = {}
for label in labels:
    column = normal[label]
    if label in SIBLINGS:
        partner, weight = SIBLINGS[label]
        column = [a + weight * b for a, b in zip(column, normal[partner])]
    final[label] = pct(column)
with open(work / 'submission.csv', 'w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(columns)
    for i, u in enumerate(test_ids):
        values = [final[label][i] for label in labels]
        assert all(math.isfinite(v) and 0.0 < v <= 1.0 for v in values)
        writer.writerow([u] + ['%.6f' % v for v in values])
receipt = {'method': 'noisyor-sibling', 'siblings': SIBLINGS, 'weights_used': {k: WEIGHTS[k] for k in parts},
           'left_out': notes, 'rows': n, 'failed_studies': {k: len(failed[k]) for k in parts},
           'failed_ids': {k: sorted(failed[k])[:20] for k in parts}, 'every_part_failed': len(every_part_failed),
           'flag_notes': flag_notes, 'flag_files': flag_files, 'r7_not_scored': r7_not_scored, 'theirs_sets': globals().get('THEIRS_DONE'),
           'theirs_seconds': round(globals().get('THEIRS_SECONDS', 0)), 'ours_end_seconds': round(globals().get('OURS_SECONDS', 0)),
           'total_seconds': round(time.time() - NB_T0)}
(work / 'blend_receipt.json').write_text(json.dumps(receipt, indent=1))
print('NOTEBOOK_COMPLETE', json.dumps(receipt), flush=True)
"""

TIMING = """# Timing only, no submission (Oct 15, 2026): the team's sets (stage 1 of the combined notebook) on 64 training studies
# laid out as a fake test set, on Kaggle's two T4s. Writes /kaggle/working/timing.json.
import csv, json, os, random, re, subprocess, sys, time
from pathlib import Path
SOURCE = __SOURCE__
N = 64
comp = next(p for p in (Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
                        Path('/kaggle/input/rsna-knee-abnormality-detection')) if (p / 'train_series.csv').is_file())
fake = Path('/kaggle/temp/fake_root')
(fake / 'test_series').mkdir(parents=True, exist_ok=True)
with open(comp / 'train_series.csv', newline='') as handle:
    reader = csv.DictReader(handle)
    fields, series = reader.fieldnames, list(reader)
studies = sorted({r['StudyInstanceUID'] for r in series})
random.Random(0).shuffle(studies)
chosen = sorted(studies[:N])
with open(fake / 'test_series.csv', 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(r for r in series if r['StudyInstanceUID'] in set(chosen))
targets = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
           'Synovitis', "Baker's", 'Contusion', 'Fracture']
with open(fake / 'test.csv', 'w', newline='') as handle:
    handle.write('StudyInstanceUID\\n' + ''.join(u + '\\n' for u in chosen))
with open(fake / 'sample_submission.csv', 'w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['StudyInstanceUID'] + targets)
    for u in chosen:
        writer.writerow([u] + [0.5] * len(targets))
for u in chosen:
    os.symlink(comp / 'train_series' / u, fake / 'test_series' / u)
out = Path('/kaggle/working/theirs')
out.mkdir(parents=True, exist_ok=True)
(out / 'team_sets.py').write_text(SOURCE, encoding='utf-8')
t0 = time.time()
rc = subprocess.run([sys.executable, '-u', str(out / 'team_sets.py')], cwd=str(out),
                    env=dict(os.environ, TEAM_DATA_ROOT=str(fake), TEAM_OUTPUT_DIR=str(out)),
                    stdout=open(out / 'log.txt', 'w'), stderr=subprocess.STDOUT).returncode
total = time.time() - t0
log = (out / 'log.txt').read_text(encoding='utf-8', errors='replace')
print(log[-6000:])
report = json.loads((out / 'six_model_inference_report.json').read_text()) if (out / 'six_model_inference_report.json').is_file() else {}
sets ={m['key']: round(m['seconds'], 1) for m in report.get('models', [])}
timing = {'studies': N, 'exit': rc, 'total_seconds': round(total, 1), 'set_seconds': sets,
          'prep_seconds': round(total - sum(sets.values()), 1) if sets else None,
          'timing_lines': re.findall(r'\\[timing\\][^\\n]*', log)}
if sets:
    per_study = (total - sum(sets.values())) / N + sum(sets.values()) / N
    timing['projected_hours_1320'] = round(per_study * 1320 / 3600, 2)
Path('/kaggle/working/timing.json').write_text(json.dumps(timing, indent=1))
print('TIMING', json.dumps(timing), flush=True)
"""


PROFILE_CREDIT = """# Knee speed profile (no submission)

Runs the three stages of the 0.966 team notebook exactly as submitted (theirs, ours, R7, in that order), each on its own
size-matched sample of 240 training studies laid out as a test set, while a sampler logs CPU, memory, disk, processes
and GPU every 5 s. The rules for reading the results were fixed beforehand. Outputs in /kaggle/working/profile.
"""

PROFILE_START = """# Profile clock, sampler and fake test folders (one per stage, disjoint, so every stage reads cold files).
import csv, json, os, subprocess, sys, time
from pathlib import Path
NB_T0 = time.time()
PROFILE = Path('/kaggle/working/profile')
PROFILE.mkdir(parents=True, exist_ok=True)
SAMPLES = __SAMPLES__


def MARK(event):
    with open(PROFILE / 'markers.jsonl', 'a') as handle:
        handle.write(json.dumps({'t': time.time(), 'event': event}) + '\\n')
    print('MARK', event, round(time.time() - NB_T0), flush=True)


(PROFILE / 'sampler.py').write_text(__SAMPLER__)
SAMPLER = subprocess.Popen([sys.executable, '-u', str(PROFILE / 'sampler.py'), str(PROFILE / 'samples.jsonl')],
                           start_new_session=True, stdout=subprocess.DEVNULL, stderr=open(PROFILE / 'sampler.err', 'w'))
_fields = 'timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu'
for _extra in (',clocks_throttle_reasons.active', ',clocks_event_reasons.active', ''):
    if subprocess.run(['nvidia-smi', '--query-gpu=' + _fields + _extra, '--format=csv'], capture_output=True).returncode == 0:
        break
GPULOG = subprocess.Popen(['nvidia-smi', '--query-gpu=' + _fields + _extra, '--format=csv', '-l', '5'],
                          stdout=open(PROFILE / 'gpu.csv', 'w'), stderr=subprocess.STDOUT, start_new_session=True)
comp = next(p for p in (Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
                        Path('/kaggle/input/rsna-knee-abnormality-detection')) if (p / 'train_series.csv').is_file())
with open(comp / 'train_series.csv', newline='') as handle:
    _reader = csv.DictReader(handle)
    _fields_series, _series = _reader.fieldnames, list(_reader)
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
           'Synovitis', "Baker's", 'Contusion', 'Fracture']
FAKE = {}
for _stage in ('theirs', 'ours', 'r7'):
    _ids = SAMPLES[_stage]
    _root = Path('/kaggle/temp/fake_' + _stage)
    (_root / 'test_series').mkdir(parents=True, exist_ok=True)
    _want = set(_ids)
    with open(_root / 'test_series.csv', 'w', newline='') as handle:
        _w = csv.DictWriter(handle, fieldnames=_fields_series)
        _w.writeheader()
        _w.writerows(r for r in _series if r['StudyInstanceUID'] in _want)
    with open(_root / 'test.csv', 'w', newline='') as handle:
        handle.write('StudyInstanceUID\\n' + ''.join(u + '\\n' for u in _ids))
    with open(_root / 'sample_submission.csv', 'w', newline='') as handle:
        _w = csv.writer(handle)
        _w.writerow(['StudyInstanceUID'] + TARGETS)
        for u in _ids:
            _w.writerow([u] + [0.5] * len(TARGETS))
    for u in _ids:
        os.symlink(comp / 'train_series' / u, _root / 'test_series' / u)
    FAKE[_stage] = _root
print('PROFILE_READY', {k: len(v) for k, v in SAMPLES.items()}, flush=True)
MARK('start')
"""

PROFILE_SAMPLER = r'''# Every 5 s: CPU, cgroup memory, disks, larger processes, R7 progress. One JSON line per sample.
import glob, json, os, sys, time
out = open(sys.argv[1], 'a')


def text(path):
    try:
        with open(path) as handle:
            return handle.read()
    except OSError:
        return None


def memory():
    d = {}
    for path in ('/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory/memory.usage_in_bytes'):
        v = text(path)
        if v:
            d['current'] = int(v)
            break
    for path in ('/sys/fs/cgroup/memory.stat', '/sys/fs/cgroup/memory/memory.stat'):
        v = text(path)
        if v:
            for line in v.splitlines():
                k, val = line.split()[:2]
                if k in ('anon', 'file', 'shmem', 'file_dirty', 'file_writeback', 'rss', 'cache', 'total_rss', 'total_cache'):
                    d[k] = int(val)
            break
    v = text('/sys/fs/cgroup/memory.events')
    if v:
        for line in v.splitlines():
            k, val = line.split()
            d['events_' + k] = int(val)
    v = text('/proc/meminfo')
    if v:
        for line in v.splitlines():
            k, rest = line.split(':')
            if k in ('MemAvailable', 'Cached', 'Shmem'):
                d['meminfo_' + k] = int(rest.split()[0]) * 1024
    return d


def processes():
    rows = []
    for p in glob.glob('/proc/[0-9]*'):
        try:
            stat = open(p + '/stat').read()
            name = stat[stat.index('(') + 1:stat.rindex(')')]
            f = stat[stat.rindex(')') + 2:].split()
            cpu, rss = int(f[11]) + int(f[12]), int(f[21]) * 4096
            if rss < 100e6 and cpu < 500:
                continue
            io = {}
            try:
                for line in open(p + '/io'):
                    k, v = line.split(':')
                    io[k] = int(v)
            except OSError:
                pass
            cmd = open(p + '/cmdline', 'rb').read().replace(b'\0', b' ')[:100].decode(errors='replace')
            rows.append([int(p[6:]), name, cpu, rss, io.get('read_bytes'), io.get('rchar'), io.get('write_bytes'), cmd])
        except (OSError, ValueError, IndexError):
            continue
    return rows


while True:
    record = {'t': time.time(), 'cpu': [int(x) for x in open('/proc/stat').readline().split()[1:9]], 'mem': memory(),
              'disk': {f[2]: [int(f[5]), int(f[9]), int(f[12])] for f in (l.split() for l in open('/proc/diskstats')) if len(f) >= 14},
              'procs': processes(), 'loadavg': open('/proc/loadavg').read().split()[:3]}
    rows = {}
    for path in glob.glob('/kaggle/temp/probe/prediction/worker*/predictions.csv') + glob.glob('/tmp/probe/prediction/worker*/predictions.csv'):
        try:
            rows[path.split('/')[-2]] = sum(1 for _ in open(path)) - 1
        except OSError:
            pass
    record['r7_rows'] = rows
    parts = glob.glob('/kaggle/working/prediction-parts/*')
    record['ours_parts'] = len(parts)
    out.write(json.dumps(record) + '\n')
    out.flush()
    time.sleep(5)
'''

PROFILE_IO = """# I/O test (about 5 minutes) on 40 more cold studies: raw reads with 1, 4 and 16 threads; pydicom in 1 and 4 processes.
import concurrent.futures, glob, json, os, time
MARK('io_start')
_io = SAMPLES['io']


def _files(ids):
    return [f for u in ids for f in sorted(glob.glob(str(comp / 'train_series' / u / '*' / '*.dcm')))]


def _read(path):
    with open(path, 'rb') as handle:
        return len(handle.read())


IO = {}
for threads, ids in ((1, _io[0:10]), (4, _io[10:20]), (16, _io[20:30])):
    files = _files(ids)
    t = time.time()
    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        size = sum(pool.map(_read, files))
    IO['read_threads_%d' % threads] = {'files': len(files), 'mb': round(size / 1e6), 'seconds': round(time.time() - t, 2),
                                       'mb_per_s': round(size / 1e6 / (time.time() - t), 1)}
_copy = _files(_io[30:40])
_dest = Path('/kaggle/temp/io_copy')
_dest.mkdir(parents=True, exist_ok=True)


def _copy_one(path):
    data = open(path, 'rb').read()
    target = _dest / (Path(path).parent.name + '_' + Path(path).name)
    target.write_bytes(data)
    return len(data)


t = time.time()
with concurrent.futures.ThreadPoolExecutor(16) as pool:
    size = sum(pool.map(_copy_one, _copy))
IO['copy_16_threads_to_temp'] = {'files': len(_copy), 'mb': round(size / 1e6), 'seconds': round(time.time() - t, 2)}
_warm = _files(_io[0:10])
_script = PROFILE / 'decode_bench.py'
_script.write_text('''import sys, time, json, pydicom
paths = [l.strip() for l in open(sys.argv[1]) if l.strip()]
t = time.time()
for p in paths:
    pydicom.dcmread(p, stop_before_pixels=True)
header = time.time() - t
t = time.time()
for p in paths:
    pydicom.dcmread(p).pixel_array
print(json.dumps({'files': len(paths), 'header_s': header, 'full_s': time.time() - t}))
''')
for procs in (1, 4):
    chunks = [_warm[i::procs] for i in range(procs)]
    lists = []
    for i, chunk in enumerate(chunks):
        path = PROFILE / ('decode_list_%d_%d.txt' % (procs, i))
        path.write_text('\\n'.join(chunk))
        lists.append(path)
    t = time.time()
    runs = [subprocess.Popen([sys.executable, str(_script), str(p)], stdout=subprocess.PIPE, text=True) for p in lists]
    results = [json.loads(r.communicate()[0]) for r in runs]
    IO['pydicom_%d_processes_warm' % procs] = {'files': len(_warm), 'wall_s': round(time.time() - t, 2), 'per_process': results}
(PROFILE / 'io.json').write_text(json.dumps(IO, indent=1))
print('IO', json.dumps(IO), flush=True)
MARK('io_end')
"""

PROFILE_END = """# Stop the sampler and collect every stage's own timing records into /kaggle/working/profile.
import shutil
MARK('end')
time.sleep(6)
for proc in (SAMPLER, GPULOG):
    try:
        os.killpg(proc.pid, 9)
    except Exception:
        pass
_work = Path('/kaggle/working')
_keep = PROFILE / 'stage-records'
_keep.mkdir(exist_ok=True)
for pattern in ('theirs/log.txt', 'theirs/six_model_inference_report.json', 'inference-summary.json', 'prediction-parts/*.json',
                'r7_receipt.json', 'worker*_log.txt'):
    for path in _work.glob(pattern):
        target = _keep / str(path.relative_to(_work)).replace('/', '__')
        shutil.copy2(path, target)
for path in Path(str(TMP)).glob('prediction/worker*/predictions.csv') if 'TMP' in globals() else []:
    shutil.copy2(path, _keep / ('r7_' + path.parent.name + '_predictions.csv'))
summary = {'stages': {'theirs_end_s': round(globals().get('THEIRS_SECONDS', 0)), 'ours_end_s': round(globals().get('OURS_SECONDS', 0)),
                      'theirs_ok': globals().get('THEIRS_OK'), 'ours_ok': globals().get('OURS_OK'), 'r7_ok': globals().get('R7_OK')},
           'total_s': round(time.time() - NB_T0), 'samples': {k: len(v) for k, v in SAMPLES.items()}}
(PROFILE / 'summary.json').write_text(json.dumps(summary, indent=1))
print('PROFILE_COMPLETE', json.dumps(summary), flush=True)
"""


def replace(text: str, old: str, new: str, count: int, what: str) -> str:
    if text.count(old) != count:
        raise ValueError(f"Expected {count} x {what}, found {text.count(old)}: {old!r}")
    return text.replace(old, new)


def newconv_block(notebook: Path = NEWCONV_NOTEBOOK) -> str:
    """Cell 1 of the teammate's newconv notebook as a set's model block: unchanged apart from its two top-level folder
    creations."""
    nb = json.loads(notebook.read_text())
    src = "".join(nb["cells"][1]["source"])
    if "Four-fold ConvNeXt-Tiny graycopy24 inference with rotation TTA" not in src:
        raise ValueError("Not the teammate's newconv cell")
    src = replace(src, "\nWORK_DIR.mkdir(parents=True, exist_ok=True)\n", "\n", 1, "newconv working folder")
    src = replace(src, "\nTEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)\n", "\n", 1, "newconv cache folder")
    compile(src, "newconv", "exec")
    return src


def their_source(notebook: Path, eff_tta: str, sets: tuple[str, ...] = SETS) -> str:
    nb = json.loads(notebook.read_text())
    if len(nb["cells"]) != 1:
        raise ValueError("Expected the 0.965 notebook to have one code cell")
    src = "".join(nb["cells"][0]["source"])
    if "RSNA: original SIX-model equal-probability ensemble" not in src:
        raise ValueError("Not the six-set (0.965) notebook")
    src = replace(src, COMPETITION_PATH,
                  "Path(__import__('os').environ.get('TEAM_DATA_ROOT', '/kaggle/input/competitions/rsna-knee-abnormality-detection'))",
                  8, "competition path")
    src = replace(src, "OUTPUT_DIR = Path('/kaggle/working')\n",
                  "OUTPUT_DIR = Path(os.environ.get('TEAM_OUTPUT_DIR', '/kaggle/working/theirs'))\n", 1, "output folder")
    src = replace(src, "    'eff': (0.0, 12.0),\n", f"    'eff': {eff_tta},\n", 1, "eff views")
    keep = (f"# Combined team notebook (Oct 15, 2026): only these sets, in this order.\n_KEEP = {sets!r}\n"
            "MODEL_SPECS = [next(s for s in MODEL_SPECS if s['key'] == k) for k in _KEEP]\n\n")
    src = replace(src, "# 下方内嵌代码分别隔离全局常量", keep + "# 下方内嵌代码分别隔离全局常量", 1, "set filter anchor")
    src = replace(src, "or len(predictions) != 6:", "or len(predictions) != len(MODEL_SPECS):", 1, "set count check")
    if "newconv" in sets:
        src = replace(src, keep, keep.replace("MODEL_SPECS = [", NEWCONV_SETUP + "MODEL_SPECS = [", 1), 1, "newconv spec")
        src = replace(src, "\ndef _digest(data):",
                      "\n_BACKEND_SOURCES['newconv'] = " + repr(newconv_block()) + "\n\n\ndef _digest(data):", 1, "newconv block")
        src = replace(src, "    if len(backend.TTA_ANGLES) > 2 or 0.0 not in backend.TTA_ANGLES:",
                      "    if (len(backend.TTA_ANGLES) > 2 and spec['key'] != 'newconv') or 0.0 not in backend.TTA_ANGLES:",
                      1, "view count check")
        src = replace(src, "    backend.FIXED_FP16_SANITY_CHECK = FP16_FIRST_BATCH_CHECK\n",
                      "    backend.FIXED_FP16_SANITY_CHECK = FP16_FIRST_BATCH_CHECK\n" + NEWCONV_HOOK, 1, "newconv hook")
    compile(src, "team_sets", "exec")
    return src


def cell(kind: str, text: str) -> dict:
    item = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def metadata(kernel_id: str, title: str, code_file: str, kind: str, datasets: list[str]) -> dict:
    return {"id": kernel_id, "title": title, "code_file": code_file, "language": "python", "kernel_type": kind,
            "is_private": True, "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": datasets, "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [],
            "model_sources": [], "docker_image_pinning_type": "original", "docker_image": DOCKER}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--their-notebook", type=Path, default=Path("notebooks/rsna-ensemble.ipynb"))
    parser.add_argument("--base-notebook", type=Path, default=Path("artifacts/kaggle-oaiseeds6/notebook/inference.ipynb"),
                        help="our six-seed + R7 notebook: cell 2 ours, cells 4-5 R7")
    parser.add_argument("--bundle-dataset", default="max6296/knee-v7oai6w56-bundle")
    parser.add_argument("--eff-tta", default="(0.0, 12.0)", choices=["(0.0, 12.0)", "(0.0,)"])
    parser.add_argument("--theirs-limit-hours", type=float, default=2.0)
    parser.add_argument("--kernel-id", default="max6296/knee-team3-lb")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--profile", action="store_true", help="build a speed-profile kernel instead (no submission)")
    parser.add_argument("--samples", type=Path, default=Path("artifacts/speed-profile/samples.json"))
    parser.add_argument("--stages", default="theirs,ours,r7", help="profile only: which stages run")
    parser.add_argument("--r7-warm", action="store_true", help="read R7's next studies ahead with many threads")
    parser.add_argument("--their-sets", default=",".join(SETS),
                        help="their sets in run order (Oct 16: conv, a teammate's 0.964 ConvNeXt-tiny, was added last)")
    parser.add_argument("--blend-weights", default="0.5,0.3,0.2",
                        help="theirs, ours, R7 (Oct 16: refitted on out-of-fold predictions to 0.4,0.35,0.25 for theirs + "
                             "conv and ours = five folds on D, our round-6 labels, plus five on the hybrid OAI labels)")
    parser.add_argument("--per-finding-weights", type=Path, default=None,
                        help="Oct 21, with --postproc noisyor-sibling: "
                             "a JSON file {finding: {theirs, ours, r7}} replacing the global weights per finding")
    parser.add_argument("--fast", action="store_true",
                        help="ours in the fast mode (persistent workers, preparing ahead, accelerate, 64-study groups); "
                             "needs a bundle exported with the Oct 16 code, otherwise the cell falls back to the submitted mode")
    parser.add_argument("--postproc", default="none", choices=["none", "noisyor-sibling"],
                        help="stage 4 (Oct 17): noisy-OR of the ranks, the sibling correction and the per-study "
                             "insurance (a study one part failed on takes the other parts' score); none = the rank blend as before")
    parser.add_argument("--hard-limits", action="store_true",
                        help="hard stops for stages 2 and 3 (Oct 20)")
    parser.add_argument("--ours-hard-hours", type=float, default=5.75, help="with --hard-limits: notebook clock")
    parser.add_argument("--r7-hard-hours", type=float, default=8.5, help="with --hard-limits: notebook clock")
    parser.add_argument("--both-layouts", action="store_true",
                        help="tell the teammates' stage where the competition data is mounted (old or new Kaggle input layout; Oct 21)")
    parser.add_argument("--early-decoders", action="store_true",
                        help="install the JPEG decoder plugins from our bundle's wheels before stage 1 (Oct 20)")
    parser.add_argument("--ours-batch-size", type=int, default=2,
                        help="studies per batch in our stage (1 for an every-window-once bundle; default 2 as submitted)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.hard_limits and (args.profile or args.timing):
        raise ValueError("--hard-limits is for the submission notebook only")
    sets = tuple(args.their_sets.split(","))
    source = their_source(args.their_notebook, args.eff_tta, sets)
    datasets = [THEIR_DATASETS[k] for k in sets]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.timing:
        slug = args.kernel_id.split("/")[1]
        (args.output_dir / "timing.py").write_text(TIMING.replace("__SOURCE__", repr(source)))
        compile((args.output_dir / "timing.py").read_text(), "timing", "exec")
        meta = metadata(args.kernel_id, slug.replace("-", " ").title(), "timing.py", "script", datasets)
        (args.output_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"wrote {args.output_dir}/timing.py")
        return
    base = json.loads(args.base_notebook.read_text())
    ours, setup, run = ("".join(base["cells"][i]["source"]) for i in (2, 4, 5))
    if "run_bundle(bundle" not in ours or "DEADLINE_HOURS = 8.3" not in setup or "T0 = NB_T0" not in setup \
            or "R7_STAGE_COMPLETE" not in run or "r7_sub.csv" not in run:
        raise ValueError("Unexpected base notebook cells")
    ours = replace(ours, "deadline_seconds=int(5.5 * 3600)",
                   "deadline_seconds=int(max(1800, 5.5 * 3600 - (time.time() - NB_T0)))", 1, "our time limit")
    if args.ours_batch_size != 2:  # Oct 20: an every-window-once bundle has a different window count per study
        ours = replace(ours, 'run_bundle(bundle, str(data_root), "/kaggle/working", "cuda", 2, 4,',
                       f'run_bundle(bundle, str(data_root), "/kaggle/working", "cuda", {args.ours_batch_size}, 4,', 1,
                       "our batch size")
    if args.fast:
        ours = replace(ours, "try:\n    run_bundle(bundle", OURS_FAST + "try:\n    run_bundle(bundle", 1, "our run call")
        ours = replace(ours, "\n               separate_processes=True)", "\n               **OURS_MODE)", 1, "our run mode")
    def r7_cell_source(setup_source):
        return R7_STAGE.replace("__SETUP__", repr(setup_source)).replace("__RUN__", repr(run)).replace(
            "__WARM_START__", R7_WARM_START if args.r7_warm else "").replace("__WARM_STOP__", R7_WARM_STOP if args.r7_warm else "")

    r7_stage = r7_cell_source(setup)
    theirs_stage = THEIRS_STAGE.replace("__SETS__", repr(sets)).replace(
        "__LIMIT__", str(int(args.theirs_limit_hours * 3600))).replace("__SOURCE__", repr(source))
    if args.profile:
        samples = {k: v["studies"] for k, v in json.loads(args.samples.read_text())["samples"].items()}
        theirs_stage = replace(theirs_stage, "env=dict(os.environ, TEAM_OUTPUT_DIR=str(THEIRS_DIR))",
                               "env=dict(os.environ, TEAM_OUTPUT_DIR=str(THEIRS_DIR), TEAM_DATA_ROOT=str(FAKE['theirs']))",
                               1, "their data folder")
        ours = replace(ours, 'data_root = next((root for root in data_roots if (root / "test.csv").is_file()), data_roots[-1])',
                       "data_root = FAKE['ours']", 1, "our data folder")
        setup = replace(setup, "DATA = find_mount(COMPETITION, 'test.csv')", "DATA = FAKE['r7']", 1, "R7 data folder")
        r7_stage = r7_cell_source(setup)  # rebuilt after the data-folder change
        stage_cells = {"theirs": cell("code", "MARK('theirs_start')\n" + theirs_stage + "MARK('theirs_end')\n"),
                       "ours": cell("code", "MARK('ours_start')\n" + OURS_STAGE.replace("__OURS__", repr(ours)) + "MARK('ours_end')\n"),
                       "r7": cell("code", "MARK('r7_start')\n" + r7_stage + "MARK('r7_end')\n")}
        cells = [cell("markdown", PROFILE_CREDIT),
                 cell("code", PROFILE_START.replace("__SAMPLES__", repr(samples)).replace("__SAMPLER__", repr(PROFILE_SAMPLER))),
                 *[stage_cells[s] for s in ("theirs", "ours", "r7") if s in args.stages.split(",")],
                 *([cell("code", PROFILE_IO)] if args.stages == "theirs,ours,r7" else []), cell("code", PROFILE_END)]
        for c in cells[1:]:
            compile("".join(c["source"]), "cell", "exec")
        compile(PROFILE_SAMPLER, "sampler", "exec")
        notebook = dict(base)
        notebook["cells"] = cells
        (args.output_dir / "inference.ipynb").write_text(json.dumps(notebook, indent=1))
        slug = args.kernel_id.split("/")[1]
        meta = metadata(args.kernel_id, slug.replace("-", " ").title(), "inference.ipynb", "notebook",
                        [args.bundle_dataset, *datasets, *R7_DATASETS])
        (args.output_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"wrote profile {args.output_dir}/inference.ipynb ({len(cells)} cells), datasets {meta['dataset_sources']}")
        return
    weights = [float(x) for x in args.blend_weights.split(",")]
    if len(weights) != 3 or abs(sum(weights) - 1) > 1e-9:
        raise ValueError(f"--blend-weights needs three weights summing to 1: {weights}")
    if not all(w > 0 for w in weights):  # Oct 17 review: a 0 weight divides by zero when only that part is left
        raise ValueError(f"--blend-weights must all be above 0 (leave a part out with its own switch): {weights}")
    blend, credit = BLEND, CREDIT
    if args.postproc == "noisyor-sibling":
        blend = BLEND_NOISYOR
        theirs_stage = replace(theirs_stage, "finally:\n    shutil.rmtree(THEIRS_CACHE, ignore_errors=True)"
                               "  # about 23 GB; ours and R7 need the disk\n", THEIRS_NO_SERIES, 1, "their cache clean-up")
        credit = replace(credit, "Stage 4 averages per-finding percentile ranks with fixed weights (theirs 0.5, ours 0.3, R7 0.2).",
                         "Stage 4 blends per-finding percentile ranks as a noisy-OR with fixed weights (theirs 0.5, ours 0.3, "
                         "R7 0.2), then applies a small fixed correction between related findings (Lateral OA and PF OA "
                         "against Medial OA, Synovitis against Effusion). A study one part failed on takes the other parts' "
                         "score.", 1, "credit")
    if weights != [0.5, 0.3, 0.2]:
        t, o, r = (f"{w:g}" for w in weights)
        if args.postproc == "none":
            blend = replace(blend, "weights fixed before any leaderboard result: theirs 0.5, ours 0.3, R7 0.2.",
                            f"weights refitted out of fold (Oct 16, 2026): theirs {t}, ours {o}, R7 {r}.", 1, "blend comment")
        else:
            blend = replace(blend, "weights theirs 0.5, ours 0.3, R7 0.2;", f"weights theirs {t}, ours {o}, R7 {r};", 1,
                            "blend comment")
            credit = replace(credit, "(theirs 0.5, ours 0.3, R7 0.2)", f"(theirs {t}, ours {o}, R7 {r})", 1, "credit weights")
        blend = replace(blend, "WEIGHTS = {'theirs': 0.5, 'ours': 0.3, 'r7': 0.2}",
                        f"WEIGHTS = {{'theirs': {t}, 'ours': {o}, 'r7': {r}}}", 1, "blend weights")
    if args.per_finding_weights is not None:
        if args.postproc != "noisyor-sibling" or weights != [0.5, 0.3, 0.2]:
            raise ValueError("--per-finding-weights needs --postproc noisyor-sibling and the default --blend-weights")
        pf = json.loads(args.per_finding_weights.read_text())
        targets = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
                   "Synovitis", "Baker's", "Contusion", "Fracture"]  # the competition's columns
        if sorted(pf) != sorted(targets) or any(sorted(v) != ["ours", "r7", "theirs"] or abs(sum(v.values()) - 1) > 1e-9
                                                or min(v.values()) <= 0 for v in pf.values()):
            raise ValueError("--per-finding-weights: every finding needs theirs, ours, r7 above 0 summing to 1")
        pf = {t: {k: pf[t][k] for k in ("theirs", "ours", "r7")} for t in targets}
        blend = replace(blend, "weights theirs 0.5, ours 0.3, R7 0.2;",
                        "weights per finding (PF_WEIGHTS, chosen out of fold on Oct 21, 2026);", 1,
                        "blend comment")
        blend = replace(blend, "WEIGHTS = {'theirs': 0.5, 'ours': 0.3, 'r7': 0.2}\n",
                        "WEIGHTS = {'theirs': 0.5, 'ours': 0.3, 'r7': 0.2}  # kept for the receipt; PF_WEIGHTS are used\n"
                        f"PF_WEIGHTS = {pf!r}\n", 1, "per-finding weights")
        blend = replace(blend, "column.append(sum(WEIGHTS[name] * terms[name][i] for name in use) / sum(WEIGHTS[name] for name in use) if use else None)",
                        "column.append(sum(PF_WEIGHTS[label][name] * terms[name][i] for name in use) / "
                        "sum(PF_WEIGHTS[label][name] for name in use) if use else None)", 1, "per-finding blend line")
        blend = replace(blend, "receipt = {'method': 'noisyor-sibling', 'siblings': SIBLINGS,",
                        "receipt = {'method': 'noisyor-sibling', 'per_finding_weights': PF_WEIGHTS, 'siblings': SIBLINGS,", 1,
                        "receipt per-finding weights")
        credit = replace(credit, "with fixed weights (theirs 0.5, ours 0.3, R7 0.2)",
                         "with fixed weights per finding (chosen out of fold, halfway between each finding's best weights and "
                         "theirs 0.5, ours 0.3, R7 0.2)", 1, "credit weights")
    ours_stage = OURS_STAGE.replace("__OURS__", repr(ours))
    if args.hard_limits:
        theirs_stage = replace(theirs_stage, "            proc.wait()\n            THEIRS_RC = 'time limit'",
                               "            try:\n                proc.wait(timeout=300)\n            except subprocess.TimeoutExpired:\n"
                               "                pass\n            THEIRS_RC = 'time limit'", 1, "their stop wait")
        ours_stage = OURS_STAGE_HARD.replace("__OURS_HARD__", str(int(args.ours_hard_hours * 3600))).replace(
            "__OURS_SCRIPT__", repr(OURS_SCRIPT.replace("__OURS__", repr(ours))))
        for command in ("['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader']",
                        "['df', '-h', '/dev/shm', '/kaggle/working', '/tmp']"):
            setup = replace(setup, f"subprocess.run({command}, check=False)", f"_bounded_run({command}, 60)", 1, "R7 setup command")
        setup = replace(setup, "subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *wheels], check=True)",
                        "_bounded_run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *wheels], 1800, check=True)",
                        1, "R7 setup pip")
        run = replace(run, "rc = subprocess.run([sys.executable, '-u', str(SRC / 'probe_dual.py'),",
                      "rc = _r7_controller([sys.executable, '-u', str(SRC / 'probe_dual.py'),", 1, "R7 controller call")
        run = replace(run, "'--max-fallback-fraction', '1.0'], env=env).returncode", "'--max-fallback-fraction', '1.0'], env)",
                      1, "R7 controller call end")
        r7_stage = replace(r7_cell_source(setup), "R7_OK = False\ntry:\n",
                           R7_HARD_HELPER.replace("__R7_HARD__", str(int(args.r7_hard_hours * 3600))) + "R7_OK = False\ntry:\n",
                           1, "R7 hard limit")
        r7_stage = replace(r7_stage, "    exec(compile(" + repr(setup) + ", 'r7_setup', 'exec'), globals())\n",
                           "    _r7_setup_bounded(compile(" + repr(setup) + ", 'r7_setup', 'exec'), R7_SETUP_SECONDS)\n", 1,
                           "R7 setup in a bounded thread")
        credit = credit.rstrip("\n") + ("\n\nHard limits (Oct 20, 2026): our stage runs as its own process and is stopped at "
                                        f"{args.ours_hard_hours:g} h, R7's controller at {args.r7_hard_hours:g} h on the "
                                        "notebook clock; a stopped stage keeps what it finished.\n")
        compile(OURS_SCRIPT.replace("__OURS__", repr(ours)), "ours_script", "exec")
    if args.both_layouts:
        # Oct 21, 2026: a Kaggle session can mount inputs in the old
        # layout (/kaggle/input/<slug>); the teammates' script hard-codes the new competition path and then stops at once.
        # Its own TEAM_DATA_ROOT setting is pointed at the folder that exists (the same path in the new layout).
        theirs_stage = replace(theirs_stage, "env=dict(os.environ, TEAM_OUTPUT_DIR=str(THEIRS_DIR)))",
                               "env=dict(os.environ, TEAM_OUTPUT_DIR=str(THEIRS_DIR), TEAM_DATA_ROOT=str(next(\n"
                               "                                    (r for r in (Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),\n"
                               "                                                 Path('/kaggle/input/rsna-knee-abnormality-detection'))\n"
                               "                                     if (r / 'test_series.csv').is_file()),\n"
                               "                                    Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')))))",
                               1, "their data folder in either layout")
    if args.early_decoders:
        theirs_stage = EARLY_DECODERS + theirs_stage
        credit = credit.rstrip("\n") + ("\n\nBefore stage 1 the notebook installs the JPEG decoder plugins (JPEG Lossless, JPEG-LS) "
                                        "from our bundle's own wheels, so every stage can read compressed DICOMs.\n")
    cells = [cell("markdown", credit), cell("code", START),
             cell("code", theirs_stage),
             cell("code", ours_stage),
             cell("code", r7_stage),
             cell("code", blend)]
    for c in cells[1:]:
        compile("".join(c["source"]), "cell", "exec")
    notebook = dict(base)
    notebook["cells"] = cells
    (args.output_dir / "inference.ipynb").write_text(json.dumps(notebook, indent=1))
    slug = args.kernel_id.split("/")[1]
    meta = metadata(args.kernel_id, slug.replace("-", " ").title().replace("Lb", "LB"), "inference.ipynb", "notebook",
                    [args.bundle_dataset, *datasets, *R7_DATASETS])
    (args.output_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.output_dir}/inference.ipynb ({len(cells)} cells), datasets {meta['dataset_sources']}")


if __name__ == "__main__":
    main()
