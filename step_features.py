from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
import pandas as pd
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.TopAbs import TopAbs_REVERSED

log = logging.getLogger(__name__)


#: Columns produced by :func:`extract_step_features` / :func:`step_features_for_paths`.
#: Named distinctly from :data:`features.FEATURE_COLUMNS`, which is the unrelated,
#: model-facing column contract further down the pipeline.
STEP_FEATURE_COLUMNS = [
    "length", "width", "height", "diagonal",
    "part_volume", "part_area",
    "estimated_stock_volume", "estimated_part_to_stock_volume_ratio",
    "n_holes", "n_slots",
]

_CLOSED_TOL = 1.0  #: degrees of slack treating an accumulated sweep as a full 360 deg ring
_HALF_TOL = 5.0  #: degrees of slack treating a sweep as a 180 deg (slot end) face
_THREAD_MIN_PATCHES = 8  #: min helical patches before a fragmented sweep counts as a thread


def _face_axis(face):
    """Classify a cylindrical/conical face by its axis.

    Returns ``None`` for faces that are neither. Otherwise returns
    ``(key, kind, radius, sweep_degrees, concave)`` where ``key`` identifies
    the face's axis (direction + foot point, rounded so faces on the same
    axis hash identically), letting the caller group faces belonging to the
    same hole/slot/thread.
    """
    adaptor = BRepAdaptor_Surface(face.wrapped)
    kind = face.geomType()
    if kind == "CYLINDER":
        surface = adaptor.Cylinder()
        radius = surface.Radius()
    elif kind == "CONE":
        surface = adaptor.Cone()
        radius = surface.RefRadius()
    else:
        return None

    axis = surface.Axis()
    origin = axis.Location()
    origin = (origin.X(), origin.Y(), origin.Z())
    direction = axis.Direction()
    direction = (direction.X(), direction.Y(), direction.Z())
    for component in direction:
        if abs(component) > 1e-9:
            if component < 0:
                direction = tuple(-c for c in direction)
            break
    offset = -sum(origin[i] * direction[i] for i in range(3))
    foot = tuple(origin[i] + offset * direction[i] for i in range(3))

    key = (tuple(round(c, 4) for c in direction), tuple(round(c, 2) for c in foot))
    sweep = math.degrees(adaptor.LastUParameter() - adaptor.FirstUParameter())
    concave = face.wrapped.Orientation() == TopAbs_REVERSED
    return key, kind, radius, sweep, concave


def count_holes_and_slots(solids) -> tuple[int, int]:
    """Estimate the number of holes and slots from cylindrical/conical faces.

    Heuristic: group each solid's cylinder/cone faces by shared axis; skip
    groups that aren't majority-concave (i.e. not pocket/hole walls). Faces
    whose swept angle sums to ~360 deg (within ``_CLOSED_TOL``) form a closed
    hole; when several radii on the same axis each close on their own (e.g. a
    counterbore), only the largest turn count is kept so a stepped hole isn't
    counted twice. A heavily fragmented sweep (>= ``_THREAD_MIN_PATCHES``
    patches) that still totals ~360 deg is treated as a threaded hole.
    Anything left with a sweep near 180 deg (within ``_HALF_TOL``) is a slot
    end cap; two such ends sharing an axis and radius make one slot.

    This is an approximation, not exact CAD feature recognition - coaxial
    distinct holes or unusual thread geometry can confuse it.
    """
    holes = slots = 0
    for solid in solids:
        axes = defaultdict(list)
        for face in solid.Faces():
            info = _face_axis(face)
            if info:
                axes[info[0]].append(info[1:])

        half_ends = defaultdict(int)
        for (direction, _), members in axes.items():
            rings = defaultdict(float)
            for kind, radius, sweep, _ in members:
                rings[(kind, round(radius, 3))] += sweep
            if sum(1 for *_, concave in members if concave) * 2 < len(members):
                continue

            turns = [round(s / 360) for s in rings.values()
                     if s >= 360 - _CLOSED_TOL
                     and abs(s - 360 * round(s / 360)) < _CLOSED_TOL]
            if turns:
                holes += max(turns)
            elif (len(members) >= _THREAD_MIN_PATCHES
                  and sum(rings.values()) >= 360 - _CLOSED_TOL):
                holes += 1
            else:
                for ring, sweep in rings.items():
                    if abs(sweep - 180) < _HALF_TOL:
                        half_ends[(direction, ring)] += 1

        slots += sum(count // 2 for count in half_ends.values())
    return holes, slots


def extract_step_features_from_workplane(wp) -> dict[str, float | bool]:
    """Compute :data:`STEP_FEATURE_COLUMNS` from an already-imported CadQuery workplane.

    Split out from :func:`extract_step_features` so callers that also need
    the parsed shape for another purpose (e.g. tessellating it for display)
    can import the STEP file once and reuse it, rather than parsing twice.
    """
    shape = wp.val()
    bb = shape.BoundingBox()

    length, width, height = sorted([bb.xlen, bb.ylen, bb.zlen], reverse=True)
    volume, area = shape.Volume(), shape.Area()
    estimated_stock_volume = length * width * height
    ratio = volume / estimated_stock_volume if estimated_stock_volume else float("nan")
    solids = wp.solids().vals() or [shape]
    n_holes, n_slots = count_holes_and_slots(solids)

    return {
        "length": length,
        "width": width,
        "height": height,
        "diagonal": bb.DiagonalLength,
        "part_volume": volume,
        "part_area": area,
        "estimated_stock_volume": estimated_stock_volume,
        "estimated_part_to_stock_volume_ratio": ratio,
        "n_holes": n_holes,
        "n_slots": n_slots,
    }


def extract_step_features(step_path: str | Path) -> dict[str, float | bool]:
    """Import a STEP file from disk and compute :data:`STEP_FEATURE_COLUMNS`."""
    import cadquery as cq

    wp = cq.importers.importStep(str(step_path))
    return extract_step_features_from_workplane(wp)


def step_features_for_paths(paths, step_dir: str | Path = ".") -> pd.DataFrame:
    """Batch-extract features for STEP files, e.g. for offline dataset building.

    A file that fails to parse contributes a row of NaNs instead of aborting
    the whole batch.
    """
    rows = []
    for p in paths:
        full = Path(step_dir) / p
        try:
            rows.append(extract_step_features(full))
        except Exception as exc:
            log.warning("STEP parse failed for %s: %s", full, exc)
            rows.append({c: float("nan") for c in STEP_FEATURE_COLUMNS})
    return pd.DataFrame(rows, columns=STEP_FEATURE_COLUMNS)
