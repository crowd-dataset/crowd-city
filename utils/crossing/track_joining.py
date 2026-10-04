"""Repair broken pedestrian tracks before any crossing logic sees them.

YOLO's tracker often splits one pedestrian into several track ids: a crossing
pedestrian hidden for a moment behind the vehicle in front of the camera comes
back under a new id, and an id can be reused much later for someone else. A
crossing rule that judges each id on its own then sees half a crossing at a
time. On a hand-reviewed busy CROWD segment, 20 of the 29 crossers YOLO
detected but the rule missed were split this way.

join_track_pieces relabels the detections so that each id is one pedestrian:
every continuous stretch of an id (no gap over RUN_GAP_SECONDS) becomes its
own piece, and a piece that starts within MAXIMUM_JOIN_GAP_SECONDS after
another ends, where that one's sideways velocity puts the pedestrian and with
a similar box size, continues it under the same id. Every later step
(detector, road-crossing rule, segmentation, speed) then works on the joined
tracks.
"""

from __future__ import annotations

import bisect
from typing import Dict, List, Tuple

import polars as pl

PERSON_CLASS = 0
# Chosen from a comparison on a hand-reviewed busy CROWD segment: sideways-only
# joining (no unique-match requirement, the default gates below) raised the
# crossers counted from 13 to 21 of 54 at precision 91% (93% without joining);
# tighter gates gave 18. On Waymo, real crossers with a track passing the
# camera rose from 162 to 182 (training) and 25 to 27 (validation).
TRACK_JOINING_ENABLED = True
# A gap longer than this within one id ends a piece (the id was reused).
RUN_GAP_SECONDS = 2.0
# Longest gap between two pieces that can still be one pedestrian.
MAXIMUM_JOIN_GAP_SECONDS = 4.0
# The follow-on piece must start within this distance of where the first
# piece's velocity puts the pedestrian: horizontally a fixed margin plus a
# share of the box height (which grows with nearness), vertically a share of
# the box height. Box heights must agree within MAXIMUM_HEIGHT_RATIO.
JOIN_X_MARGIN = 0.06
JOIN_X_HEIGHT_SHARE = 0.5
JOIN_Y_HEIGHT_SHARE = 0.6
MAXIMUM_HEIGHT_RATIO = 1.5
# Link only when exactly one piece fits the gates; in a crowd, another person
# standing where the lost one is expected is a common source of wrong links.
REQUIRE_UNIQUE_MATCH = False
# Only pedestrians walking sideways are joined (the case that matters for
# crossings): both pieces must move in the same direction at least
# MINIMUM_SIDEWAYS_SPEED (image widths per second), at speeds within
# MAXIMUM_SPEED_RATIO of each other.
REQUIRE_SIDEWAYS_MOTION = True
MINIMUM_SIDEWAYS_SPEED = 0.03
MAXIMUM_SPEED_RATIO = 2.0
# Seconds at the end of a piece over which its sideways velocity is measured.
VELOCITY_SECONDS = 1.0
# Recorded in the results.pickle fingerprint; bump when joining changes.
TRACK_JOINING_VERSION = "track_joining_sideways_v1"


def _pieces(people: pl.DataFrame, fps: float) -> List[Dict]:
    """Split every person id into continuous pieces, with their end states."""
    maximum_gap = max(1, int(round(RUN_GAP_SECONDS * fps)))
    velocity_frames = max(1, int(round(VELOCITY_SECONDS * fps)))
    pieces: List[Dict] = []
    for track in people.partition_by("unique-id", maintain_order=True):
        track = track.sort("frame-count").unique(subset=["frame-count"], keep="first", maintain_order=True)
        frames = track.get_column("frame-count").to_list()
        x = track.get_column("x-center").to_list()
        y = track.get_column("y-center").to_list()
        height = track.get_column("height").to_list()
        identifier = track.get_column("unique-id")[0]
        start = 0
        for index in range(1, len(frames) + 1):
            if index < len(frames) and frames[index] - frames[index - 1] <= maximum_gap:
                continue
            last = index - 1
            first_for_velocity = start
            while frames[last] - frames[first_for_velocity] > velocity_frames:
                first_for_velocity += 1
            span = frames[last] - frames[first_for_velocity]
            last_for_velocity = start
            while last_for_velocity < last and frames[last_for_velocity + 1] - frames[start] <= velocity_frames:
                last_for_velocity += 1
            start_span = frames[last_for_velocity] - frames[start]
            pieces.append({
                "id": identifier,
                "first": frames[start],
                "last": frames[last],
                "start_state": (x[start], y[start], height[start]),
                "end_state": (x[last], y[last], height[last]),
                "velocity": (x[last] - x[first_for_velocity]) / span if span > 0 else 0.0,
                "start_velocity": (x[last_for_velocity] - x[start]) / start_span if start_span > 0 else None,
            })
            start = index
    return pieces


def _links(pieces: List[Dict], fps: float) -> Dict[int, int]:
    """Return {piece: following piece}, choosing the closest match for each piece end."""
    maximum_gap = MAXIMUM_JOIN_GAP_SECONDS * fps
    minimum_speed = MINIMUM_SIDEWAYS_SPEED / fps
    order = sorted(range(len(pieces)), key=lambda i: pieces[i]["first"])
    starts = [pieces[i]["first"] for i in order]
    following: Dict[int, int] = {}
    taken = set()
    for i in sorted(range(len(pieces)), key=lambda i: pieces[i]["last"]):
        piece = pieces[i]
        if REQUIRE_SIDEWAYS_MOTION and abs(piece["velocity"]) < minimum_speed:
            continue
        end_x, end_y, end_height = piece["end_state"]
        best, best_score, fitting = None, None, 0
        low = bisect.bisect_right(starts, piece["last"])
        high = bisect.bisect_right(starts, piece["last"] + maximum_gap)
        for position in range(low, high):
            j = order[position]
            if j in taken or j == i:
                continue
            candidate = pieces[j]
            if REQUIRE_SIDEWAYS_MOTION:
                following_speed = candidate["start_velocity"]
                if following_speed is not None and following_speed * piece["velocity"] < 0:
                    continue
                if following_speed is not None and abs(following_speed) >= minimum_speed:
                    speed_ratio = abs(following_speed) / abs(piece["velocity"])
                    if not 1.0 / MAXIMUM_SPEED_RATIO <= speed_ratio <= MAXIMUM_SPEED_RATIO:
                        continue
                elif following_speed is not None:
                    continue
            gap = candidate["first"] - piece["last"]
            start_x, start_y, start_height = candidate["start_state"]
            ratio = start_height / end_height if end_height > 0 else float("inf")
            if not 1.0 / MAXIMUM_HEIGHT_RATIO <= ratio <= MAXIMUM_HEIGHT_RATIO:
                continue
            dx = abs(start_x - (end_x + piece["velocity"] * gap))
            dy = abs(start_y - end_y)
            if dx > JOIN_X_MARGIN + JOIN_X_HEIGHT_SHARE * end_height or dy > JOIN_Y_HEIGHT_SHARE * end_height:
                continue
            score = dx + dy + 0.1 * abs(1.0 - ratio)
            fitting += 1
            if best_score is None or score < best_score:
                best, best_score = j, score
        if best is not None and (fitting == 1 or not REQUIRE_UNIQUE_MATCH):
            following[i] = best
            taken.add(best)
    return following


def join_track_pieces(detections: pl.DataFrame, fps: float) -> pl.DataFrame:
    """Return ``detections`` with person track ids split into pieces and pieces joined.

    Rows of other classes keep their id unless the same id also carries the
    person's frames, in which case they follow the person piece that covers
    their frame.
    """
    required = {"unique-id", "frame-count", "yolo-id", "x-center", "y-center", "height"}
    if not TRACK_JOINING_ENABLED or detections.height == 0 or not required.issubset(detections.columns) or fps <= 0:
        return detections
    people = detections.filter((pl.col("yolo-id") == PERSON_CLASS) & pl.col("unique-id").is_not_null())
    if people.height == 0:
        return detections
    pieces = _pieces(people, fps)
    following = _links(pieces, fps)

    # New ids: the first piece of every original id keeps it; later pieces get
    # fresh ids above every existing one. A joined chain takes its head's id.
    id_dtype = detections.schema["unique-id"]
    textual = id_dtype == pl.Utf8
    next_id = int(float(detections.get_column("unique-id").cast(pl.Float64, strict=False).max() or 0)) + 1
    seen = set()
    own_id: List = []
    for piece in pieces:
        if piece["id"] in seen:
            own_id.append(str(next_id) if textual else next_id)
            next_id += 1
        else:
            own_id.append(piece["id"])
            seen.add(piece["id"])
    previous = {j: i for i, j in following.items()}
    chain_id = list(own_id)
    for i in range(len(pieces)):
        if i in previous:
            continue
        current = i
        while current in following:
            current = following[current]
            chain_id[current] = own_id[i]

    table = pl.DataFrame(
        {
            "unique-id": [piece["id"] for piece in pieces],
            "_first": [piece["first"] for piece in pieces],
            "_last": [piece["last"] for piece in pieces],
            "_joined": chain_id,
        },
        schema_overrides={"unique-id": id_dtype, "_joined": id_dtype},
    )
    if all(a == b for a, b in zip(chain_id, (piece["id"] for piece in pieces))):
        return detections

    indexed = detections.with_row_index("_row")
    matched = (
        indexed.join(table, on="unique-id", how="inner")
        .filter(pl.col("frame-count").is_between(pl.col("_first"), pl.col("_last")))
        .select(["_row", "_joined"])
        .unique(subset=["_row"], keep="first")
    )
    return (
        indexed.join(matched, on="_row", how="left")
        .with_columns(pl.coalesce([pl.col("_joined"), pl.col("unique-id")]).alias("unique-id"))
        .drop(["_row", "_joined"])
    )


def joining_summary(before: pl.DataFrame, after: pl.DataFrame) -> Tuple[int, int]:
    """Person ids before and after joining, for logging."""
    def count(frame: pl.DataFrame) -> int:
        return frame.filter(pl.col("yolo-id") == PERSON_CLASS).get_column("unique-id").n_unique()

    return count(before), count(after)
