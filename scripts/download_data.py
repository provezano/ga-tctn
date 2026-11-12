"""Download the CLUTO text collections (re0, re1, tr11-tr45, wap, fbis, hitech, ...).

The original server (glaros.dtc.umn.edu) no longer serves the archive, so this
fetches the Internet Archive copy. Only the plain-text .mat/.rclass/.clabel
members are extracted.

    python -I scripts/download_data.py data/            # all collections used by the paper
    python -I scripts/download_data.py data/ re0 tr45   # a subset
"""

from __future__ import annotations

import hashlib
import os
import sys
import tarfile
import urllib.request

URL = ("http://web.archive.org/web/20170305183159id_/"
       "http://glaros.dtc.umn.edu/gkhome/fetch/sw/cluto/datasets.tar.gz")
SHA256 = "dbecf52dab217a5372159fe50a190dd1a0cb51c970d1f85c1c52aafd346e9de0"
PAPER_SETS = ["fbis", "hitech", "re0", "re1", "tr11", "tr12", "tr23", "tr31", "tr41", "tr45", "wap"]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    out = argv[0]
    wanted = argv[1:] or PAPER_SETS
    os.makedirs(out, exist_ok=True)
    archive = os.path.join(out, "cluto_datasets.tar.gz")
    if not os.path.exists(archive):
        print(f"downloading {URL}", file=sys.stderr)
        urllib.request.urlretrieve(URL, archive)
    with open(archive, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if digest != SHA256:
        print(f"checksum mismatch ({digest}); delete {archive} and retry", file=sys.stderr)
        return 1
    names = {f"{w}.mat{suffix}" for w in wanted for suffix in ("", ".rclass", ".clabel")}
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and os.path.basename(m.name) in names]
        for m in members:
            m.name = os.path.basename(m.name)
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(out, members=members, **kwargs)
    found = sorted(m.name for m in members)
    print("\n".join(found))
    missing = sorted(names - set(found))
    if missing:
        print("not in archive:", ", ".join(missing), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
