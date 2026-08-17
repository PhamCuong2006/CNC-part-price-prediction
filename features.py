from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd

TARGET_COLUMN: Final[str] = "price"

MATERIAL_FAMILIES: Final[tuple[str, ...]] = (
    "Aluminum",
    "Carbon Steel",
    "Plastic",
    "Stainless Steel",
    "Titanium",
)

MATERIAL_TO_FAMILY: Final[dict[str, str]] = {
    "301 SST": "Stainless Steel",
    "303 SST": "Stainless Steel",
    "304 SST": "Stainless Steel",
    "316 SST": "Stainless Steel",
    "416 PH": "Stainless Steel",
    "420 PH": "Stainless Steel",
    "4140": "Carbon Steel",
    "CRS": "Carbon Steel",
    "HRS": "Carbon Steel",
    "O1": "Carbon Steel",
    "STEEL": "Carbon Steel",
    "ALUM 5052-H32": "Aluminum",
    "ALUM 5052-T6": "Aluminum",
    "ALUM 6061": "Aluminum",
    "ALUM 6061-T6": "Aluminum",
    "ALUM 7075-T6": "Aluminum",
    "ALUM MIC-6": "Aluminum",
    "ALUMINUM": "Aluminum",
    "ACETAL": "Plastic",
    "DELRIN": "Plastic",
    "NYLON 6/6": "Plastic",
    "PEEK": "Plastic",
    "POLYCARBONATE": "Plastic",
    "PVC": "Plastic",
    "ULTEM 1000": "Plastic",
    "TITANIUM": "Titanium",
}

#: Columns supplied by STEP geometry extraction.
GEOMETRY_COLUMNS: Final[tuple[str, ...]] = (
    "length",
    "diagonal",
    "part_volume",
    "estimated_stock_volume",
    "estimated_part_to_stock_volume_ratio",
    "n_holes",
    "n_slots",
)

#: Columns supplied by the operator through the form.
MANUAL_COLUMNS: Final[tuple[str, ...]] = (
    "material_family",
    "complexity_score",
    "tolerances",
    "quantity",
    "estimated_machining_time",
    "setup_count",
    "tool_changes_count",
)

#: Raw columns consumed by :func:`build_features`, before engineering.
RAW_COLUMNS: Final[tuple[str, ...]] = (
    "material_family",
    "complexity_score",
    "tolerances",
    "length",
    "part_volume",
    "estimated_stock_volume",
    "estimated_part_to_stock_volume_ratio",
    "n_slots",
    "n_holes",
    "estimated_machining_time",
    "quantity",
    "setup_count",
    "tool_changes_count",
    "diagonal",
)

#: Exact column order the fitted model expects. XGBoost validates feature names
#: on predict, so this ordering is part of the model contract.
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "material_family",
    "complexity_score",
    "tolerances",
    "length",
    "estimated_part_to_stock_volume_ratio",
    "n_slots",
    "n_holes",
    "estimated_machining_time",
    "quantity",
    "diagonal",
    "machining_overhead",
    "tool_complexity",
    "log_part_volume",
    "log_stock_volume",
)

_DROPPED_AFTER_ENGINEERING: Final[tuple[str, ...]] = (
    "part_volume",
    "estimated_stock_volume",
    "setup_count",
    "tool_changes_count",
)


class FeatureError(ValueError):
    """Raised when a record cannot be turned into a valid feature row."""


def build_features(data: pd.DataFrame, categories: Sequence[str] = MATERIAL_FAMILIES,) -> pd.DataFrame:
    """Turn raw operator/geometry columns into the model's engineered feature frame.

    Raises :class:`FeatureError` if a required column is missing, if
    ``material_family`` is null for any row, or if it holds a value outside
    ``categories``.
    """
    missing = [column for column in RAW_COLUMNS if column not in data.columns]
    if missing:
        raise FeatureError(f"Missing required columns: {', '.join(missing)}")

    levels = list(categories)
    features = data.loc[:, list(RAW_COLUMNS)].copy()

    n_missing_family = int(features["material_family"].isna().sum())
    if n_missing_family:
        raise FeatureError(
            f"material_family is missing for {n_missing_family} row(s); "
            f"every row must specify one of: {', '.join(levels)}"
        )

    unknown = sorted(set(features["material_family"].unique()) - set(levels))
    if unknown:
        raise FeatureError(
            f"Unknown material_family value(s): {', '.join(map(str, unknown))}. "
            f"Expected one of: {', '.join(levels)}"
        )

    features["machining_overhead"] = features["estimated_machining_time"] * features["setup_count"]
    features["tool_complexity"] = (
        features["estimated_machining_time"] * features["tool_changes_count"]
    )
    features["log_part_volume"] = np.log1p(features["part_volume"])
    features["log_stock_volume"] = np.log1p(features["estimated_stock_volume"])

    features = features.drop(columns=list(_DROPPED_AFTER_ENGINEERING))
    features["material_family"] = pd.Categorical(features["material_family"], categories=levels)

    return features.loc[:, list(FEATURE_COLUMNS)]


def extract_target(data: pd.DataFrame) -> pd.Series:
    """Return the ``price`` column, raising :class:`FeatureError` if absent."""
    if TARGET_COLUMN not in data.columns:
        raise FeatureError(f"Missing target column {TARGET_COLUMN!r}")
    return data[TARGET_COLUMN]


def build_single_sample(record: dict[str, float | str], categories: Sequence[str] = MATERIAL_FAMILIES) -> pd.DataFrame:
    """Build a one-row feature frame for a single prediction request."""
    return build_features(pd.DataFrame([record]), categories=categories)