#!/usr/bin/env python3
"""Download selected files of an NDA (NIMH Data Archive) package (e.g. OAI image series), resumable and size-checked.

Credentials come from the NDA_USERNAME / NDA_PASSWORD environment variables (never written to disk). The selection CSV
needs package_file_id, download_alias and file_size columns; files land at <out>/<download_alias>. Files already present
with the right size are skipped; partial files are written as *.partial and renamed only when complete. NDA lists some
files with a placeholder size of 1 byte; those are checked against the size the server sends instead.
"""
from __future__ import annotations

import argparse
import csv
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

LOGIN = "https://nda.nih.gov/api/ras/user/login"
API = "https://nda.nih.gov/api/package"


class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.token = None
        self.login()

    def login(self):
        with self.lock:
            r = requests.post(LOGIN, auth=(os.environ["NDA_USERNAME"], os.environ["NDA_PASSWORD"]), timeout=60)
            r.raise_for_status()
            self.token = r.text.strip()

    def presigned(self, package, ids):
        for attempt in range(5):
            r = requests.post(f"{API}/{package}/files/batchGeneratePresignedUrls", json=list(ids), timeout=300,
                              headers={"Authorization": f"Bearer {self.token}", "Accept": "application/json"})
            if r.status_code == 401:
                self.login()
                continue
            if r.ok:
                return {e["package_file_id"]: e["downloadURL"] for e in r.json()["presignedUrls"]}
            time.sleep(10 * (attempt + 1))
        r.raise_for_status()


def fetch(url, target: Path, size: int) -> int:
    partial = target.with_name(target.name + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        if size <= 1:  # placeholder size in the NDA listing
            size = int(r.headers["Content-Length"])
        with open(partial, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if partial.stat().st_size != size:
        partial.unlink()
        raise IOError(f"size mismatch for {target}")
    partial.rename(target)
    return size


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=int, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--batch", type=int, default=500)
    args = p.parse_args()
    rows = list(csv.DictReader(open(args.selection)))
    def present(r):  # complete files are only ever renamed into place, so a placeholder-size file that exists is done
        path = args.out / r["download_alias"]
        return path.exists() and (int(r["file_size"]) <= 1 or path.stat().st_size == int(r["file_size"]))
    todo = [r for r in rows if not present(r)]
    total = sum(int(r["file_size"]) for r in todo)
    print(f"{len(rows)} selected, {len(rows) - len(todo)} already present, {len(todo)} to fetch ({total / 1e9:.1f} GB)", flush=True)
    session = Session()
    done_bytes, failed, start = 0, [], time.time()
    for i in range(0, len(todo), args.batch):
        batch = todo[i:i + args.batch]
        urls = session.presigned(args.package, [int(r["package_file_id"]) for r in batch])
        with ThreadPoolExecutor(args.threads) as pool:
            jobs = {pool.submit(fetch, urls[int(r["package_file_id"])], args.out / r["download_alias"], int(r["file_size"])): r
                    for r in batch}
            for job in as_completed(jobs):
                try:
                    done_bytes += job.result()
                except Exception as e:  # retried on the next run
                    failed.append(jobs[job]["download_alias"])
                    print(f"failed {jobs[job]['download_alias']}: {e}", flush=True)
        rate = done_bytes / max(time.time() - start, 1) / 1e6
        print(f"{min(i + args.batch, len(todo))}/{len(todo)} files, {done_bytes / 1e9:.1f} GB, {rate:.0f} MB/s, "
              f"{len(failed)} failed", flush=True)
    print("finished" if not failed else f"finished with {len(failed)} failures; run again to retry", flush=True)


if __name__ == "__main__":
    main()
