"""Persistent store for derived road-surface labels.

Only the derived index is retained, never the masks: for every crossing track
the store holds one row per sampled frame giving the surface under the
pedestrian's footpoint. That keeps the store small enough to sit beside the
Parquet detection store.

A track's labels depend only on the video, the track's own boxes and the
segmentation settings (since v4 they no longer depend on which other tracks
share a window). So each stored track carries a signature of its boxes, and a
label is reused exactly when a request's track has the same signature under
the same settings, whichever crossing configuration or program asked for it.
The store is shared in this format with crowd-jaywalking
(scripts/crossing/crowd_city_surface_store.py), so a segment either project has
segmented is not segmented again by the other. Keep the two in step:
INDEX_SCHEMA, the settings keys, track_signature and canonical_track_id must
match exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import polars as pl

import common
from custom_logger import CustomLogger


logger = CustomLogger(__name__)

# v2: detection frames are mapped to video time with the video's probed frame
# rate instead of the integer rate in the file name. v1 labels were read under
# boxes that drift by one frame every ~33 s on 29.97 fps footage, so they are
# re-segmented rather than reused.
# v3: merged windows longer than the per-window ceiling are split instead of
# dropped, so v2 entries can be missing whole stretches of crossing tracks.
# v4: frames are sampled on a fixed grid of video time, so a track's labels
# no longer depend on which other tracks share its window.
# v5: tracks are validated one by one by a signature of their boxes instead of
# a fingerprint of the crossing configuration, ids are canonical, and new
# tracks are merged into a segment's entry instead of replacing it. Shared
# with crowd-jaywalking. v4 entries made by this pipeline stay readable (see
# LEGACY_INDEX_SCHEMA) and are rewritten as v5 when next used.
INDEX_SCHEMA = "crowd_surface_index_v5"
LEGACY_INDEX_SCHEMA = "crowd_surface_index_v4"
INDEX_FOLDER = "index"
MANIFEST_FOLDER = "manifest"
URL_CACHE_FILE = "video_urls.json"

INDEX_COLUMNS = {
    "unique-id": pl.Utf8,
    "frame-count": pl.Int64,
    "video_time_s": pl.Float64,
    "surface": pl.Utf8,
    "surface_code": pl.Int8,
    "confidence": pl.Float32,
}


def configured_segmentation_root() -> Optional[str]:
    """Return the configured segmentation store root, if any."""
    try:
        value = common.get_configs("seg_data")
    except Exception:
        return None
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        root = os.fspath(value)
    elif isinstance(value, Sequence):
        roots = [os.fspath(item) for item in value if str(item).strip()]
        if not roots:
            return None
        if len(roots) > 1:
            logger.warning(
                "More than one 'seg_data' root is configured; using the first "
                f"({roots[0]}). Detection file names are globally unique, so a "
                "single store is sufficient."
            )
        root = roots[0]
    else:
        return None
    return root if str(root).strip() else None


@dataclass(frozen=True)
class SegmentationSettings:
    """Everything that changes the meaning of a stored surface label."""

    model_identifier: str
    coarse_hz: float
    refine_hz: float
    footpoint_band_fraction: float
    footpoint_width_fraction: float
    minimum_confidence: float
    # Only used to recognise v4 entries, which were keyed by it. A v5 label
    # does not depend on the crossing configuration, only on its track.
    crossing_fingerprint: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_identifier": self.model_identifier,
            "coarse_hz": float(self.coarse_hz),
            "refine_hz": float(self.refine_hz),
            "footpoint_band_fraction": float(self.footpoint_band_fraction),
            "footpoint_width_fraction": float(self.footpoint_width_fraction),
            "minimum_confidence": float(self.minimum_confidence),
        }

    def legacy_dict(self) -> Dict[str, Any]:
        """The settings block a v4 entry made under these settings carries."""
        return {**self.as_dict(), "crossing_fingerprint": self.crossing_fingerprint}


def crossing_fingerprint(parameters: Dict[str, Any]) -> str:
    """Return a stable hash of the crossing configuration."""
    payload = json.dumps(parameters, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def canonical_track_id(value: Any) -> str:
    """Return the form a track id is stored under: 2118, 2118.0 and "2118" are one id."""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def track_signature(track: pl.DataFrame) -> str:
    """Return the identity of a track's boxes; a stored label is reused only for the same boxes.

    Frames and centres are what the footpoint sampling reads, rounded so that
    the same detection read through different paths (CSV, Parquet, a list of
    observations) gives the same signature.
    """
    ordered = track.sort("frame-count")
    digest = hashlib.sha256()
    for part in (
        ordered.get_column("frame-count").cast(pl.Int64, strict=False).to_list(),
        ordered.get_column("x-center").cast(pl.Float64, strict=False).round(4).to_list(),
        ordered.get_column("y-center").cast(pl.Float64, strict=False).round(4).to_list(),
    ):
        digest.update(json.dumps(part).encode("utf-8"))
    return digest.hexdigest()[:16]


class SurfaceStore:
    """Read and write the derived surface index."""

    def __init__(self, root: str) -> None:
        self.root = Path(root).expanduser()
        self.index_dir = self.root / INDEX_FOLDER
        self.manifest_dir = self.root / MANIFEST_FOLDER
        self._url_cache: Optional[Dict[str, str]] = None

    def ensure_directories(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        """Return whether the store root exists or can be created."""
        try:
            self.ensure_directories()
        except OSError as error:
            logger.warning(f"Segmentation store {self.root} is unavailable: {error}")
            return False
        return True

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def index_path(self, stem: str) -> Path:
        return self.index_dir / f"{stem}.parquet"

    def manifest_path(self, stem: str) -> Path:
        return self.manifest_dir / f"{stem}.json"

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    def read_manifest(self, stem: str) -> Optional[Dict[str, Any]]:
        path = self.manifest_path(stem)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError) as error:
            logger.warning(f"Unreadable segmentation manifest {path}: {error}")
            return None

    def load_tracks(
        self,
        stem: str,
        settings: SegmentationSettings,
        signatures: Dict[str, str],
        accept_legacy: bool = True,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], bool]:
        """Return the stored rows of every requested track the store covers.

        ``signatures`` maps canonical track ids to track_signature. A v5 entry
        covers a track when the settings match and the stored signature is the
        same. A v4 entry made by this pipeline covers it, as v4 did, when the
        settings and the crossing fingerprint match and the id is listed. The
        second value says whether a v4 entry was used, so it can be rewritten
        as v5. Tracks not covered are simply absent from the result.
        """
        manifest = self.read_manifest(stem)
        if manifest is None:
            return {}, False
        schema = manifest.get("schema")
        if schema == INDEX_SCHEMA and manifest.get("settings") == settings.as_dict():
            stored = manifest.get("track_signatures") or {}
            covered = {
                track_id for track_id, signature in signatures.items()
                if stored.get(track_id) == signature
            }
            legacy = False
        elif (
            accept_legacy
            and schema == LEGACY_INDEX_SCHEMA
            and settings.crossing_fingerprint
            and manifest.get("settings") == settings.legacy_dict()
        ):
            listed = {canonical_track_id(value) for value in manifest.get("track_ids") or []}
            covered = set(signatures) & listed
            legacy = True
        else:
            return {}, False
        if not covered:
            return {}, False

        index = self.read_index(stem)
        if index is None:
            return {}, False
        rows: Dict[str, List[Dict[str, Any]]] = {}
        for row in index.iter_rows(named=True):
            track_id = canonical_track_id(row["unique-id"])
            if track_id in covered:
                rows.setdefault(track_id, []).append({**row, "unique-id": track_id})
        return rows, legacy

    def save_tracks(
        self,
        stem: str,
        settings: SegmentationSettings,
        signatures: Dict[str, str],
        rows: List[Dict[str, Any]],
        extra: Optional[Dict[str, Any]] = None,
        producer: str = "crowd-city",
    ) -> None:
        """Store the rows of the tracks in ``signatures``, keeping the other tracks of the entry.

        Tracks of an existing v5 entry under the same settings that this call
        does not mention are kept with their rows; a track it mentions is
        replaced. An entry in any other schema or settings is replaced whole.
        """
        tracks = {canonical_track_id(key): value for key, value in signatures.items()}
        rows = [{**row, "unique-id": canonical_track_id(row["unique-id"])} for row in rows]
        previous = self.read_manifest(stem) or {}
        kept_signatures: Dict[str, str] = {}
        kept_rows: List[Dict[str, Any]] = []
        if previous.get("schema") == INDEX_SCHEMA and previous.get("settings") == settings.as_dict():
            kept_signatures = {
                track_id: signature
                for track_id, signature in (previous.get("track_signatures") or {}).items()
                if track_id not in tracks
            }
            index = self.read_index(stem)
            if index is not None and kept_signatures:
                kept_rows = [
                    row for row in index.iter_rows(named=True)
                    if canonical_track_id(row["unique-id"]) in kept_signatures
                ]
        merged_signatures = {**kept_signatures, **tracks}

        self.write_index(stem, kept_rows + rows)
        # Facts about the segment (video id, start, frame rates) are kept from
        # the previous entry unless this call states them again.
        manifest = {
            key: value for key, value in previous.items()
            if key not in {"schema", "settings", "track_ids", "track_signatures", "producer"}
        }
        manifest.update(extra or {})
        manifest.update({
            "schema": INDEX_SCHEMA,
            "producer": producer,
            "settings": settings.as_dict(),
            "track_ids": sorted(merged_signatures),
            "track_signatures": dict(sorted(merged_signatures.items())),
        })
        self._write_manifest(stem, manifest)

    def merge_entry(self, stem: str, other: "SurfaceStore") -> Tuple[bool, bool]:
        """Add the tracks ``other`` holds for ``stem`` that this store lacks.

        Used to share one store between machines (sync_surface_store.py) and
        with crowd-jaywalking. Only v5 entries under the same settings are
        merged; this store's version of a track it already holds is kept. A
        missing or pre-v5 entry here is replaced by the other v5 entry. Returns
        whether this store changed, and whether it holds tracks (or versions
        of tracks) that ``other`` lacks.
        """
        theirs = other.read_manifest(stem) or {}
        ours = self.read_manifest(stem) or {}
        ours_is_current = ours.get("schema") == INDEX_SCHEMA
        if theirs.get("schema") != INDEX_SCHEMA:
            return False, ours_is_current
        if not ours_is_current:
            index = other.read_index(stem)
            if index is None:
                return False, False
            self.write_index(stem, index.to_dicts())
            self._write_manifest(stem, theirs)
            return True, False
        if ours.get("settings") != theirs.get("settings"):
            return False, False

        our_tracks = dict(ours.get("track_signatures") or {})
        their_tracks = dict(theirs.get("track_signatures") or {})
        take = {track_id: signature for track_id, signature in their_tracks.items() if track_id not in our_tracks}
        ours_extra = any(their_tracks.get(track_id) != signature for track_id, signature in our_tracks.items())
        if not take:
            return False, ours_extra

        our_index = self.read_index(stem)
        their_index = other.read_index(stem)
        if their_index is None:
            return False, ours_extra
        rows = [] if our_index is None else [
            row for row in our_index.iter_rows(named=True)
            if canonical_track_id(row["unique-id"]) in our_tracks
        ]
        rows += [
            {**row, "unique-id": canonical_track_id(row["unique-id"])}
            for row in their_index.iter_rows(named=True)
            if canonical_track_id(row["unique-id"]) in take
        ]
        merged_tracks = dict(sorted({**our_tracks, **take}.items()))
        manifest = {**theirs, **ours, "track_ids": sorted(merged_tracks), "track_signatures": merged_tracks}
        self.write_index(stem, rows)
        self._write_manifest(stem, manifest)
        return True, ours_extra

    def _write_manifest(self, stem: str, manifest: Dict[str, Any]) -> None:
        path = self.manifest_path(stem)
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        temporary.replace(path)

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------
    def read_index(self, stem: str) -> Optional[pl.DataFrame]:
        path = self.index_path(stem)
        if not path.is_file():
            return None
        try:
            return pl.read_parquet(path)
        except Exception as error:
            logger.warning(f"Unreadable surface index {path}: {error}")
            return None

    def write_index(self, stem: str, rows: List[Dict[str, Any]]) -> None:
        frame = (
            pl.DataFrame(rows, schema=INDEX_COLUMNS)
            if rows
            else pl.DataFrame(schema=INDEX_COLUMNS)
        )
        path = self.index_path(stem)
        temporary = path.with_suffix(".parquet.tmp")
        frame.write_parquet(temporary, compression="zstd")
        temporary.replace(path)

    # ------------------------------------------------------------------
    # Resolved video URLs
    # ------------------------------------------------------------------
    def _url_cache_path(self) -> Path:
        return self.root / URL_CACHE_FILE

    def cached_url(self, video_id: str) -> Optional[str]:
        if self._url_cache is None:
            path = self._url_cache_path()
            if path.is_file():
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        self._url_cache = dict(json.load(handle))
                except (OSError, ValueError):
                    self._url_cache = {}
            else:
                self._url_cache = {}
        return self._url_cache.get(str(video_id))

    def store_url(self, video_id: str, url: str) -> None:
        if self._url_cache is None:
            self.cached_url(video_id)
        assert self._url_cache is not None
        self._url_cache[str(video_id)] = str(url)
        path = self._url_cache_path()
        temporary = path.with_suffix(".json.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(self._url_cache, handle, indent=2, sort_keys=True)
            temporary.replace(path)
        except OSError as error:
            logger.warning(f"Could not persist resolved video URLs: {error}")
