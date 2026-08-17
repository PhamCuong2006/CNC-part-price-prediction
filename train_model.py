from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from features import FEATURE_COLUMNS, build_features, extract_target

log = logging.getLogger(__name__)

MODEL_FILENAME: Final[str] = "model.json"
METADATA_FILENAME: Final[str] = "metadata.json"

RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.2
PRICE_CAP_QUANTILE: Final[float] = 0.97

#: Columns whose observed range is recorded so the app can flag inputs that
#: fall outside anything the model was trained on.
RANGE_COLUMNS: Final[tuple[str, ...]] = (
    "price",
    "length",
    "diagonal",
    "part_volume",
    "estimated_stock_volume",
    "estimated_part_to_stock_volume_ratio",
    "n_holes",
    "n_slots",
    "complexity_score",
    "tolerances",
    "quantity",
    "estimated_machining_time",
    "setup_count",
    "tool_changes_count",
)

MODEL_PARAMS: Final[dict[str, object]] = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 4,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.285,
    "gamma": 0.005,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "enable_categorical": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


@dataclass(frozen=True, slots=True)
class Metrics:
    mae: float
    rmse: float
    mape: float
    r2: float
    weighted_mape: float


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Metrics:
    """Compute MAE/RMSE/MAPE/R2 plus a value-weighted MAPE across all rows."""
    total = float(np.sum(np.asarray(y_true)))
    weighted = (
        float(np.sum(np.abs(np.asarray(y_true) - y_pred)) / total * 100) if total else float("nan")
    )
    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(root_mean_squared_error(y_true, y_pred)),
        mape=float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        r2=float(r2_score(y_true, y_pred)),
        weighted_mape=weighted,
    )


def train(data: pd.DataFrame) -> tuple[XGBRegressor, dict[str, object]]:
    """Fit the price model on ``data`` and return it with training metadata."""
    categories = tuple(sorted(data["material_family"].dropna().unique()))
    log.info("Material family levels: %s", ", ".join(categories))

    features = build_features(data, categories=categories)
    target = extract_target(data)

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    price_cap = float(y_train.quantile(PRICE_CAP_QUANTILE))
    y_train_log = np.log1p(np.minimum(y_train, price_cap))

    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(x_train, y_train_log)

    train_pred = np.expm1(model.predict(x_train))
    test_pred = np.expm1(model.predict(x_test))

    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    ranges = {
        column: {"min": float(data[column].min()), "max": float(data[column].max())}
        for column in RANGE_COLUMNS
        if column in data.columns
    }

    metadata: dict[str, object] = {
        "model_file": MODEL_FILENAME,
        "feature_columns": list(FEATURE_COLUMNS),
        "material_families": list(categories),
        "training_ranges": ranges,
        "price_cap": price_cap,
        "target_transform": "log1p",
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "train_metrics": asdict(train_metrics),
        "test_metrics": asdict(test_metrics),
        "xgboost_version": xgboost.__version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return model, metadata


def report(metadata: dict[str, object]) -> None:
    for split in ("train", "test"):
        metrics = metadata[f"{split}_metrics"]
        assert isinstance(metrics, dict)
        label = split.capitalize()
        print(f"{label} MAE           : ${metrics['mae']:.2f}")
        print(f"{label} RMSE          : ${metrics['rmse']:.2f}")
        print(f"{label} MAPE          : {metrics['mape']:.2f}%")
        print(f"{label} R2            : {metrics['r2']:.4f}")
        print(f"{label} weighted MAPE : {metrics['weighted_mape']:.3f}%")
        print()


def save(model: XGBRegressor, metadata: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / MODEL_FILENAME)
    (output_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("Wrote artifacts to %s", output_dir.resolve())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    if not args.data.is_file():
        log.error("Dataset not found: %s", args.data)
        return 1

    data = pd.read_csv(args.data)
    log.info("Loaded %d rows from %s", len(data), args.data)

    model, metadata = train(data)
    report(metadata)
    save(model, metadata, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())