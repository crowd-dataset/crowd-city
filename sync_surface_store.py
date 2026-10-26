"""Share the road-surface label store with another machine and with crowd-jaywalking.

Both projects keep their labels in the same format (utils/segmentation/store.py, schema v5) and meet in
one shared folder on a server. Entries are merged, never overwritten: a segment's entry collects the
tracks either side has segmented, so neither segments them again.

    python sync_surface_store.py pull --remote "$SHARED" --key ~/.ssh/id_mesh
    python sync_surface_store.py push --remote "$SHARED" --key ~/.ssh/id_mesh

with, for the CROWD machines, SHARED=salam@100.80.227.26:/media/salam/crowd-tue-3/pedestrians_in-youtube/seg
(lin4080's drive, where crowd-jaywalking also syncs).

``pull`` adds the shared folder's tracks to the local store (``seg_data`` in config); run it before an
analysis, not during one, as it writes to the local store. ``push`` adds the local tracks to the shared
folder and never writes the local store, so it is safe while an analysis is running. A copy of the shared
folder is kept next to the local store (``<store>.shared``) so that each sync transfers only what changed.
Transfers use rsync over ssh (``--key`` for the identity file).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from utils.segmentation.store import INDEX_FOLDER, MANIFEST_FOLDER, SurfaceStore, configured_segmentation_root


def _rsync(source: str, target: str, key: str, files: List[str] | None = None) -> None:
    command = ["rsync", "-a", "-e", f"ssh -o BatchMode=yes -i {key}" if key else "ssh -o BatchMode=yes"]
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as listing:
        if files is not None:
            listing.write("\n".join(files) + "\n")
            listing.flush()
            command += ["--files-from", listing.name]
        command += [source.rstrip("/") + "/", target.rstrip("/") + "/"]
        subprocess.run(command, check=True)


def _stems(store: SurfaceStore) -> List[str]:
    return sorted(
        path.stem for path in store.manifest_dir.glob("*.json")
        if (store.index_dir / f"{path.stem}.parquet").is_file()
    )


def refresh_mirror(remote: str, mirror: SurfaceStore, key: str) -> None:
    """Bring the local copy of the shared folder up to date (nothing is deleted)."""
    mirror.ensure_directories()
    _rsync(remote, str(mirror.root), key)


def pull(local: SurfaceStore, mirror: SurfaceStore) -> int:
    changed = 0
    for stem in _stems(mirror):
        updated, _ = local.merge_entry(stem, mirror)
        changed += int(updated)
    return changed


def push(remote: str, local: SurfaceStore, mirror: SurfaceStore, key: str) -> int:
    changed = []
    for stem in _stems(local):
        updated, _ = mirror.merge_entry(stem, local)
        if updated:
            changed.append(stem)
    if changed:
        # Index files first, then manifests, so a manifest never names tracks its index lacks.
        _rsync(str(mirror.root), remote, key, [f"{INDEX_FOLDER}/{stem}.parquet" for stem in changed])
        _rsync(str(mirror.root), remote, key, [f"{MANIFEST_FOLDER}/{stem}.json" for stem in changed])
    return len(changed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("direction", choices=["pull", "push"])
    parser.add_argument("--remote", required=True, help="user@host:/folder of the shared store")
    parser.add_argument("--local", default=None, help="local store (default: seg_data in config)")
    parser.add_argument("--key", default="", help="ssh identity file")
    arguments = parser.parse_args()

    root = arguments.local or configured_segmentation_root()
    if not root:
        print("No local store: set seg_data in config or pass --local.", file=sys.stderr)
        return 1
    local = SurfaceStore(root)
    local.ensure_directories()
    mirror = SurfaceStore(str(Path(root).expanduser()).rstrip("/") + ".shared")

    refresh_mirror(arguments.remote, mirror, arguments.key)
    if arguments.direction == "pull":
        count = pull(local, mirror)
        print(f"pull: {count} local entries gained tracks from {arguments.remote}")
    else:
        count = push(arguments.remote, local, mirror, arguments.key)
        print(f"push: {count} entries of {arguments.remote} gained tracks from {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
