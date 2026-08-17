from __future__ import annotations

import functools
import logging
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from step_features import extract_step_features_from_workplane

log = logging.getLogger(__name__)

STEP_SUFFIXES: Final[frozenset[str]] = frozenset({".step", ".stp"})
STEP_MAGIC: Final[str] = "ISO-10303-21"
MAX_UPLOAD_BYTES: Final[int] = 100 * 1024 * 1024
_HEADER_BYTES: Final[int] = 8192
_TESSELLATION_TOLERANCE: Final[float] = 0.1

#: Application protocols that describe machined mechanical parts.
MECHANICAL_SCHEMAS: Final[tuple[str, ...]] = (
    "CONFIG_CONTROL_DESIGN",
    "AUTOMOTIVE_DESIGN",
    "AP203",
    "AP214",
    "AP242",
    "MECHANICAL_DESIGN",
    "INTEGRATED_CAD",
)

#: Recognised non-mechanical protocols, mapped to a human-readable domain.
NON_MECHANICAL_SCHEMAS: Final[dict[str, str]] = {
    "ELECTRONIC_ASSEMBLY_INTERCONNECT_AND_PACKAGING_DESIGN": "electronics (AP210)",
    "ELECTROTECHNICAL": "electrotechnical plants (AP212)",
    "PLANT_SPATIAL_CONFIGURATION": "process plants (AP227)",
    "SHIP_": "ship structures (AP215/216/218)",
    "BUILDING_": "building construction",
}

_FILE_SCHEMA_RE: Final[re.Pattern[str]] = re.compile(
    r"FILE_SCHEMA\s*\(\s*\((?P<body>.*?)\)\s*\)", re.IGNORECASE | re.DOTALL
)


class StepValidationError(ValueError):
    """Raised when an upload is not a usable mechanical STEP part."""


@dataclass(frozen=True, slots=True)
class PartGeometry:
    """Engineered features plus the detected STEP application protocol."""

    features: dict[str, float]
    schema: str


@dataclass(frozen=True, slots=True)
class PartMesh:
    """A triangulated display mesh for 3D preview."""

    vertices: np.ndarray
    faces: np.ndarray

    @property
    def triangle_count(self) -> int:
        """Number of triangles in the display mesh."""
        return int(self.faces.shape[0])


def _decode_header(payload: bytes) -> str:
    """Decode the leading bytes of a STEP file for header inspection.

    Uppercased so schema matching is case-insensitive regardless of how the
    exporting CAD system cased its ``FILE_SCHEMA`` entry.
    """
    return payload[:_HEADER_BYTES].decode("latin-1", errors="replace").upper()


def _extract_schema(header: str) -> str:
    """Return the first quoted identifier in the header's ``FILE_SCHEMA`` entry."""
    match = _FILE_SCHEMA_RE.search(header)
    if not match:
        return ""
    body = match.group("body")
    names = re.findall(r"'([^']*)'", body)
    return names[0].strip() if names else ""


def validate_upload(filename: str, payload: bytes) -> str:
    """Check that an upload is a usable mechanical STEP part.

    Validates extension, size, STEP magic header and application protocol
    without invoking the CAD kernel. Returns the detected schema string on
    success, or raises :class:`StepValidationError` with a user-facing
    explanation of what's wrong.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in STEP_SUFFIXES:
        raise StepValidationError(
            f"'{filename}' is not a STEP file. Expected a .step or .stp file, "
            f"but got '{suffix or 'no extension'}'."
        )

    if not payload:
        raise StepValidationError(f"'{filename}' is empty.")

    if len(payload) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise StepValidationError(
            f"'{filename}' is {len(payload) / 1024 / 1024:.1f} MB, which exceeds "
            f"the {limit_mb} MB limit."
        )

    header = _decode_header(payload)
    if STEP_MAGIC not in header:
        raise StepValidationError(
            f"'{filename}' has a .step extension but is not a valid STEP file - "
            f"the required '{STEP_MAGIC};' header is missing. The file may be "
            f"corrupted or simply renamed from another format."
        )

    schema = _extract_schema(header)
    if not schema:
        raise StepValidationError(
            f"'{filename}' declares no FILE_SCHEMA, so its application protocol "
            f"cannot be identified. Re-export it as AP203, AP214 or AP242."
        )

    for marker, domain in NON_MECHANICAL_SCHEMAS.items():
        if marker in schema:
            raise StepValidationError(
                f"'{filename}' uses the '{schema}' protocol, which describes "
                f"{domain}, not a machined mechanical part. This model can only "
                f"price mechanical parts exported as AP203, AP214 or AP242."
            )

    if not any(marker in schema for marker in MECHANICAL_SCHEMAS):
        raise StepValidationError(
            f"'{filename}' uses the unsupported '{schema}' protocol. Re-export "
            f"the part as AP203, AP214 or AP242."
        )

    return schema


def _tessellate(shape) -> tuple[np.ndarray, np.ndarray]:
    """Tessellate an already-parsed shape into a display mesh."""
    import cadquery as cq

    if not isinstance(shape, cq.Shape):
        raise StepValidationError(
            "The STEP file contains no solid shape, only construction geometry."
        )
    raw_vertices, raw_faces = shape.tessellate(_TESSELLATION_TOLERANCE)

    if not raw_faces:
        raise StepValidationError(
            "The STEP file contains no renderable surfaces. It may hold only "
            "construction geometry such as points, axes or sketches."
        )

    vertices = np.array([(v.x, v.y, v.z) for v in raw_vertices], dtype=np.float32)
    faces = np.array(raw_faces, dtype=np.int32)
    return vertices, faces


@contextmanager
def _staged(filename: str, payload: bytes) -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmpdir:
        step_path = Path(tmpdir) / f"upload{Path(filename).suffix.lower()}"
        step_path.write_bytes(payload)
        yield step_path


@functools.lru_cache(maxsize=8)
def _parsed_workplane(filename: str, payload: bytes):
    """Import a validated STEP upload into a CadQuery workplane, once.

    Cached (by filename + raw bytes) so that :func:`load_geometry` and
    :func:`load_mesh` - which need the same parsed shape for two different
    purposes - don't each re-run the CAD kernel over the same upload.
    Failed parses are not cached, matching the previous behaviour.
    """
    import cadquery as cq

    with _staged(filename, payload) as step_path:
        return cq.importers.importStep(str(step_path))


def _safe_parse(filename: str, payload: bytes):
    """`_parsed_workplane`, translating any parse failure into a friendly error."""
    try:
        return _parsed_workplane(filename, payload)
    except Exception as exc:  # noqa: BLE001 - OCCT raises bare Exceptions
        log.warning("STEP parse failed for %s: %s", filename, exc)
        raise StepValidationError(
            f"'{filename}' could not be read as a solid model. The geometry "
            f"kernel reported: {exc}"
        ) from exc


def load_geometry(filename: str, payload: bytes) -> PartGeometry:
    """Validate an uploaded STEP file and extract its engineered features."""
    schema = validate_upload(filename, payload)
    wp = _safe_parse(filename, payload)
    raw_features = extract_step_features_from_workplane(wp)

    volume = float(raw_features.get("part_volume", 0.0) or 0.0)
    if not np.isfinite(volume) or volume <= 0.0:
        raise StepValidationError(
            f"'{filename}' contains no solid body with positive volume. "
            f"Surface- or wireframe-only models cannot be priced, because "
            f"the model needs part volume and stock volume."
        )

    features = {key: float(value) for key, value in raw_features.items()}
    return PartGeometry(features=features, schema=schema)


def load_mesh(filename: str, payload: bytes) -> PartMesh:
    """Validate an uploaded STEP file and tessellate it for 3D display."""
    validate_upload(filename, payload)
    wp = _safe_parse(filename, payload)
    vertices, faces = _tessellate(wp.val())
    return PartMesh(vertices=vertices, faces=faces)