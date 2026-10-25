"""Download only the authorized three-sequence follow-up archive manifest."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nda_tools_deps'))
import NDATools

old_home = Path(NDATools.NDA_ORGINIZATION_ROOT_FOLDER).parent
progress_root = ROOT / 'oai/tool_state_followup_redownload'
progress_root.mkdir(parents=True, exist_ok=True)
for name, value in list(vars(NDATools).items()):
    if name.startswith('NDA_') and isinstance(value, str):
        try:
            relative = Path(value).relative_to(old_home)
        except ValueError:
            continue
        # A fresh progress namespace is required after the archives were deleted.
        # The NDA downloader otherwise trusts old completion logs without checking disk.
        setattr(NDATools, name, str(progress_root / relative))
NDATools._get_keyring = False
NDATools._set_keyring = False

from NDATools.clientscripts.downloadcmd import parse_args
from NDATools.Download import Download

import os
_package_file = os.environ.get('NDA_PACKAGE_MAP')
if not _package_file:
    raise SystemExit('Set NDA_PACKAGE_MAP to your own V01/V03/V05/V06 package JSON; see REPRODUCE.md')
PACKAGES = json.loads(Path(_package_file).read_text(encoding='utf-8'))
if set(PACKAGES) != {'V01','V03','V05','V06'} or not all(isinstance(v,int) and v>0 for v in PACKAGES.values()):
    raise ValueError('NDA package map must contain positive package IDs for all four visits')
STATUS = ROOT / 'oai/score_audit/followup_MRI_download_status.json'

def main():
    import shutil
    destination = Path(r'D:\OAI\followup_6250_three_sequences')
    existing = destination
    while not existing.exists():
        existing = existing.parent
    remaining = 153388869421
    if destination.exists():
        remaining -= sum(p.stat().st_size for p in destination.rglob('*.tar.gz'))
    if shutil.disk_usage(existing).free < max(remaining, 0) + 20000000000:
        raise SystemExit('Insufficient disk space: require archives plus 20 GB reserve.')
    destination.mkdir(parents=True, exist_ok=True)
    print('DOWNLOAD: 6250 exams, 18798 archives, 153.39 GB; three sequences; 8 workers.', flush=True)
    import os
    username = os.environ.get('NDA_USERNAME') or input('NDA username: ').strip()
    if not username:
        raise SystemExit('Username required.')
    config = None
    state = {'stage': 'awaiting_authentication', 'completed_visits': [], 'packages': PACKAGES,
             'destination': str(destination), 'expected_archives': 18798, 'expected_bytes': 153388869421}
    STATUS.write_text(json.dumps(state, indent=2), encoding='utf-8')
    for visit, package in PACKAGES.items():
        dest = destination / visit
        dest.mkdir(parents=True, exist_ok=True)
        sys.argv = ['downloadcmd', '-dp', str(package), '-u', username,
                    '-d', str(dest), '-t', str(ROOT / 'oai/score_audit/followup_deduplicated' / f'{visit}_three_sequence_s3_links.txt'), '-wt', '8']
        args = parse_args()
        if config is None:
            config = NDATools.init_and_create_configuration(args, NDATools.NDA_TOOLS_DOWNLOADCMD_LOGS_FOLDER)
        state['stage'] = 'downloading'
        state['current_visit'] = visit
        STATUS.write_text(json.dumps(state, indent=2), encoding='utf-8')
        print(f'\nDownloading selected MRI for {visit}: personal package {package}', flush=True)
        Download(config, args).start()
        state['completed_visits'].append(visit)
        STATUS.write_text(json.dumps(state, indent=2), encoding='utf-8')
    state['stage'] = 'download_command_completed'
    STATUS.write_text(json.dumps(state, indent=2), encoding='utf-8')
    print('\nDownload commands finished. Confirm all archive counts and sizes before reporting completion.', flush=True)

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        print('Selected follow-up download stopped. Completed archives are retained for resume.', flush=True)
        raise
