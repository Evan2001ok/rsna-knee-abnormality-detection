"""Run the official NDA verifier with all working files inside this project."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "nda_tools_deps"))

import NDATools

# Keep NDA's settings, logs and manifests within this workspace.
home_root = Path(NDATools.NDA_ORGINIZATION_ROOT_FOLDER).parent
for name, value in list(vars(NDATools).items()):
    if name.startswith("NDA_") and isinstance(value, str):
        try:
            relative = Path(value).relative_to(home_root)
        except ValueError:
            continue
        setattr(NDATools, name, str(root / "oai" / "tool_state" / relative))
(root / "oai" / "tool_state").mkdir(parents=True, exist_ok=True)

# Prompt locally each time; do not persist the password in the credential store.
NDATools._get_keyring = False
NDATools._set_keyring = False

from NDATools.clientscripts.downloadcmd import main

if "--help" in sys.argv or "-h" in sys.argv:
    sys.argv = ["downloadcmd", "--help"]
else:
    image_download = "--images" in sys.argv
    workers = int(next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--workers=")), "4"))
    destination = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--destination=")), None)
    scoring_package = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--scores=")), None)
    metadata_only = "--metadata" in sys.argv
    package = scoring_package or next((arg.split('=',1)[1] for arg in sys.argv if arg.startswith('--package=')), '1249823')
    print("OAI package: " + package)
    print("DOWNLOAD SELECTED MRI: 2,399 knees, 12,053 archives, 137.58 GB." if image_download else ("DOWNLOAD SCORING TABLES AND DOCUMENTATION; no MRI images." if scoring_package else ("DOWNLOAD METADATA ONLY: about 92 MB; no MRI images." if metadata_only else "VERIFY ONLY: creates the file list; does not download MRI images.")))
    if image_download:
        import shutil
        if not destination:
            raise SystemExit("MRI download requires an explicit --destination=PATH.")
        target = Path(destination).resolve()
        existing = target
        while not existing.exists():
            existing = existing.parent
        completed_bytes = sum(p.stat().st_size for p in target.rglob("*.tar.gz")) if target.exists() else 0
        if shutil.disk_usage(existing).free < max(0, 137_581_517_952 - completed_bytes) + 10_000_000_000:
            raise SystemExit("Insufficient free space: require selected archives plus 10 GB reserve.")
        target.mkdir(parents=True, exist_ok=True)
        print("Destination: " + str(target))
    username = sys.argv[1].strip() if len(sys.argv) > 1 else input("NDA username (from NDA Profile): ").strip()
    if not username:
        raise SystemExit("A username is required.")
    sys.argv = [
        "downloadcmd", "-dp", package, "-u", username,
        "-d", str(target if image_download else root / "oai" / ("scores" if scoring_package else "baseline")),
    ]
    if image_download:
        sys.argv += ["-t", str(root / "oai" / "score_audit" / "multisequence_candidate_s3_links.txt"), "-wt", str(workers)]
    elif scoring_package:
        sys.argv += ["-wt", "4"]
    elif metadata_only:
        # Obtain metadata from the caller's own package; do not require a private
        # pre-exported list of subject archive links in the public source tree.
        sys.argv += ['--file-regex', r'(^|/)(image03|package_info|dataset_collection)\.txt$', '-wt', '4']
    else:
        sys.argv += ["--verify"]
main()
