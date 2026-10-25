"""Download only official image03 tables and package inventories; never MRI archives."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nda_tools_deps'))
import NDATools

old_home = Path(NDATools.NDA_ORGINIZATION_ROOT_FOLDER).parent
for name, value in list(vars(NDATools).items()):
    if name.startswith('NDA_') and isinstance(value, str):
        try:
            relative = Path(value).relative_to(old_home)
        except ValueError:
            continue
        setattr(NDATools, name, str(ROOT / 'oai/tool_state' / relative))
NDATools._get_keyring = False
NDATools._set_keyring = False

from NDATools.clientscripts.downloadcmd import parse_args
from NDATools.Download import Download

package_map = os.environ.get('NDA_PACKAGE_MAP')
if not package_map:
    raise SystemExit('Set NDA_PACKAGE_MAP to your own V01/V03/V05/V06 package-map JSON.')
PACKAGES = json.loads(Path(package_map).read_text(encoding='utf-8'))
if set(PACKAGES) != {'V01','V03','V05','V06'} or any(type(v) is not int or v <= 0 for v in PACKAGES.values()):
    raise SystemExit('Package map must contain positive integer package IDs for V01/V03/V05/V06.')
PACKAGES = {k:PACKAGES[k] for k in ('V01','V03','V05','V06')}
STATUS = ROOT / 'oai/score_audit/followup_index_download_status.json'
STATUS.parent.mkdir(parents=True, exist_ok=True)

def main():
    print('OAI FOLLOW-UP INDEX DOWNLOAD ONLY. No MRI .tar.gz files will be downloaded.', flush=True)
    username = os.environ.get('NDA_USERNAME','').strip() or input('NDA username (from Profile): ').strip()
    if not username:
        raise SystemExit('Username required.')
    config = None
    state = {'stage': 'running', 'completed_visits': [], 'packages': PACKAGES}
    STATUS.write_text(json.dumps(state, indent=2), encoding='utf-8')
    for visit, package in PACKAGES.items():
        dest = ROOT / 'oai/followup_indices' / visit
        dest.mkdir(parents=True, exist_ok=True)
        sys.argv = ['downloadcmd', '-dp', str(package), '-u', username,
                    '-d', str(dest), '--file-regex', r'(^|/)(image03|package_info|dataset_collection)\.txt$', '-wt', '2']
        args = parse_args()
        if config is None:
            config = NDATools.init_and_create_configuration(args, NDATools.NDA_TOOLS_DOWNLOADCMD_LOGS_FOLDER)
        print(f'\nDownloading index for {visit}: personal package {package}', flush=True)
        Download(config, args).start()
        state['completed_visits'].append(visit)
        STATUS.write_text(json.dumps(state, indent=2), encoding='utf-8')
    state['stage'] = 'download_command_completed'
    STATUS.write_text(json.dumps(state, indent=2), encoding='utf-8')
    print('\nIndex commands finished. Tables still require matching and completeness checks.', flush=True)

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        print('Index download stopped; no bulk MRI download was requested.', flush=True)
        raise
