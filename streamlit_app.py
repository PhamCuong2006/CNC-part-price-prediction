from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBRegressor

from features import (
    GEOMETRY_COLUMNS,
    MATERIAL_TO_FAMILY,
    FeatureError,
    build_single_sample,
)
from geometry import PartGeometry, PartMesh, StepValidationError, load_geometry, load_mesh

log = logging.getLogger(__name__)

ARTIFACT_DIR: Final[Path] = Path(__file__).parent / "artifacts"
METADATA_PATH: Final[Path] = ARTIFACT_DIR / "metadata.json"

TOLERANCE_PRESETS: Final[tuple[float, ...]] = (0.02, 0.05, 0.13, 0.25, 0.38, 2.54)
YES_NO: Final[tuple[str, ...]] = ("No", "Yes")

_STYLE: Final[str] = """
<style>
  .price-readout {
      border: 1px solid rgba(120, 130, 145, 0.35);
      border-left: 4px solid #2f6f4e;
      border-radius: 4px;
      padding: 1.1rem 1.4rem;
      margin-top: 0.4rem;
  }
  .price-readout .label {
      font-size: 0.72rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      opacity: 0.65;
  }
  .price-readout .value {
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 2.9rem;
      font-weight: 600;
      line-height: 1.15;
  }
  .price-readout .band {
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 0.85rem;
      opacity: 0.75;
  }
</style>
"""


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[XGBRegressor, dict[str, Any]]:
    """Load the trained model and its metadata, cached for the app's lifetime."""
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"No model artifacts in {ARTIFACT_DIR}. Run 'python train_model.py' first."
        )
    metadata: dict[str, Any] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    model = XGBRegressor(enable_categorical=True)
    model.load_model(ARTIFACT_DIR / str(metadata["model_file"]))
    return model, metadata


@st.cache_data(show_spinner=False, max_entries=8)
def cached_geometry(filename: str, payload: bytes) -> PartGeometry:
    return load_geometry(filename, payload)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_mesh(filename: str, payload: bytes) -> PartMesh:
    return load_mesh(filename, payload)

def mesh_figure(mesh: PartMesh) -> go.Figure:
    """Build a Plotly 3D mesh figure from a tessellated part."""
    vertices, faces = mesh.vertices, mesh.faces
    figure = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#9fb2c4",
                opacity=1.0,
                flatshading=True,
                lighting={"ambient": 0.55, "diffuse": 0.85, "specular": 0.25},
                hoverinfo="skip",
            )
        ]
    )
    figure.update_layout(
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "X (mm)"},
            "yaxis": {"title": "Y (mm)"},
            "zaxis": {"title": "Z (mm)"},
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=520,
        showlegend=False,
    )
    return figure


@st.dialog("3D preview", width="large")
def preview_dialog(filename: str, payload: bytes, geometry: PartGeometry) -> None:
    """Show a rotatable 3D mesh of the uploaded part with headline dimensions."""
    try:
        with st.spinner("Building the mesh..."):
            mesh = cached_mesh(filename, payload)
    except StepValidationError as exc:
        st.error(str(exc), icon=":material/error:")
        return

    st.plotly_chart(mesh_figure(mesh), width='stretch')
    dims = geometry.features
    left, middle, right = st.columns(3)
    left.metric("Bounding box", f"{dims['length']:.1f} x {dims['width']:.1f} x {dims['height']:.1f} mm")
    middle.metric("Part volume", f"{dims['part_volume'] / 1000:.1f} cm3")
    right.metric("Holes / slots", f"{int(dims['n_holes'])} / {int(dims['n_slots'])}")
    st.caption(f"{mesh.triangle_count:,} triangles - drag to rotate, scroll to zoom.")

def collect_inputs(metadata: dict[str, Any]) -> dict[str, Any]:
    """Render the process-input form and return the collected values."""
    families: list[str] = list(metadata["material_families"])
    materials = sorted(MATERIAL_TO_FAMILY)

    st.subheader("Process inputs")

    row1 = st.columns(3)
    material = row1[0].selectbox("Material", materials, index=materials.index("ALUM 6061-T6"))
    default_family = MATERIAL_TO_FAMILY.get(material, families[0])
    material_family = row1[1].selectbox(
        "Material family",
        families,
        index=families.index(default_family),
        help="Set automatically from the material grade. This is the field the model reads.",
    )
    tolerances = row1[2].number_input(
        "Tolerances (mm)",
        min_value=0.001,
        max_value=25.0,
        value=0.25,
        step=0.01,
        format="%.3f",
        help=f"Typical values: {', '.join(str(t) for t in TOLERANCE_PRESETS)}",
    )

    row2 = st.columns(3)
    complexity_score = row2[0].number_input(
        "Complexity score", min_value=1, max_value=10, value=5, step=1
    )
    quantity = row2[1].number_input("Quantity", min_value=1, max_value=10_000, value=1, step=1)
    estimated_machining_time = row2[2].number_input(
        "Estimated machining time (h)",
        min_value=0.001,
        max_value=100.0,
        value=0.25,
        step=0.05,
        format="%.4f",
    )

    row3 = st.columns(3)
    setup_count = row3[0].number_input("Setup count", min_value=1, max_value=20, value=2, step=1)
    tool_changes_count = row3[1].number_input(
        "Tool changes count", min_value=1, max_value=100, value=4, step=1
    )
    special_type = row3[2].selectbox("Special type", YES_NO, index=0)

    specialized_tool = st.columns(3)[0].selectbox("Specialized tool", YES_NO, index=0)

    return {
        "material": material,
        "material_family": material_family,
        "tolerances": float(tolerances),
        "special_type": special_type,
        "complexity_score": int(complexity_score),
        "specialized_tool": specialized_tool,
        "quantity": int(quantity),
        "estimated_machining_time": float(estimated_machining_time),
        "setup_count": int(setup_count),
        "tool_changes_count": int(tool_changes_count),
    }


def out_of_range_fields(sample: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    """List human-readable warnings for sample fields outside the training data's range."""
    ranges: dict[str, dict[str, float]] = metadata.get("training_ranges", {})
    warnings: list[str] = []
    for field, bounds in ranges.items():
        if field == "price" or field not in sample:
            continue
        value = float(sample[field])
        if value < bounds["min"] or value > bounds["max"]:
            warnings.append(
                f"{field.replace('_', ' ')} = {value:,.4g} "
                f"(trained on {bounds['min']:,.4g} to {bounds['max']:,.4g})"
            )
    return warnings

def predict_price(model: XGBRegressor, sample: dict[str, Any], metadata: dict[str, Any]) -> float:
    """Score one sample and invert the model's log1p price transform."""
    frame = build_single_sample(sample, categories=metadata["material_families"])
    prediction_log = model.predict(frame)
    return float(np.expm1(prediction_log)[0])


def format_value(value: Any) -> str:
    """Render a scored-sample value for display, without losing tiny nonzero floats."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | np.integer):
        return f"{int(value):,}"
    if isinstance(value, float | np.floating):
        number = float(value)
        if not np.isfinite(number):
            return str(number)
        if number.is_integer() and abs(number) < 1e15:
            return f"{int(number):,}"
        if number != 0.0 and round(number, 4) == 0.0:
            return f"{number:.2e}"
        return f"{number:,.4f}".rstrip("0").rstrip(".")
    return str(value)


def render_result(price: float, sample: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Render the predicted price, range warnings, and a scored-sample breakdown."""
    mae = float(metadata["test_metrics"]["mae"])
    st.markdown(
        f"""
        <div class="price-readout">
          <div class="label">Predicted price</div>
          <div class="value">${price:,.2f}</div>
          <div class="band">typical error +/- ${mae:,.0f} on held-out parts</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for warning in out_of_range_fields(sample, metadata):
        st.warning(f"Outside training range: {warning}", icon=":material/warning:")

    with st.expander("Scored sample"):
        used = set(GEOMETRY_COLUMNS) | {
            "material_family",
            "complexity_score",
            "tolerances",
            "quantity",
            "estimated_machining_time",
            "setup_count",
            "tool_changes_count",
        }
        table = pd.DataFrame(
            [
                {
                    "Field": key,
                    "Value": format_value(value),
                    "Used by model": "yes" if key in used else "no",
                }
                for key, value in sample.items()
            ]
        ).astype(str)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(
            "Fields marked 'no' are recorded with the quote but are not model "
            "inputs: material grade is represented by its family, and special "
            "type and specialized tool have a single value throughout the "
            "training data, so they carry no signal."
        )

def main() -> None:
    """Render the page."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    st.set_page_config(
        page_title="CNC part price estimator",
        page_icon=":material/precision_manufacturing:",
        layout="wide",
    )
    st.markdown(_STYLE, unsafe_allow_html=True)

    st.title("CNC part price estimator")
    st.caption("Upload a STEP model, add the process inputs, and generate a price.")

    try:
        model, metadata = load_model()
    except FileNotFoundError as exc:
        st.error(str(exc), icon=":material/error:")
        st.stop()

    upload_column, action_column = st.columns([3, 1], vertical_alignment="center")

    with upload_column:
        upload = st.file_uploader(
            "STEP file",
            type=["step", "stp"],
            help="Mechanical part exported as AP203, AP214 or AP242.",
        )

    geometry: PartGeometry | None = None
    upload_name = upload.name if upload is not None else ""
    if upload is not None:
        try:
            with st.spinner("Reading geometry..."):
                geometry = cached_geometry(upload_name, upload.getvalue())
        except StepValidationError as exc:
            st.error(str(exc), icon=":material/error:")
        except Exception as exc:  # noqa: BLE001 - surface anything OCCT throws
            st.error(f"Could not read '{upload_name}': {exc}", icon=":material/error:")

    with action_column:
        generate = st.button(
            "Generate price", type="primary", width='stretch', icon=":material/sell:"
        )
        view = st.button(
            "View in 3D",
            width='stretch',
            disabled=geometry is None,
            icon=":material/view_in_ar:",
        )

    if geometry is not None:
        protocol = geometry.schema.split("{")[0].strip()
        st.success(
            f"Loaded {upload_name} - {geometry.features['length']:.1f} x "
            f"{geometry.features['width']:.1f} x {geometry.features['height']:.1f} mm, "
            f"{int(geometry.features['n_holes'])} holes, "
            f"{int(geometry.features['n_slots'])} slots. "
            f"Protocol: {protocol}.",
            icon=":material/check_circle:",
        )

    if view and geometry is not None and upload is not None:
        preview_dialog(upload_name, upload.getvalue(), geometry)

    st.divider()
    inputs = collect_inputs(metadata)
    st.divider()

    if generate:
        if geometry is None:
            st.error(
                "Upload a valid STEP file before generating a price.",
                icon=":material/error:",
            )
            return

        sample: dict[str, Any] = {
            **{key: geometry.features[key] for key in GEOMETRY_COLUMNS},
            **inputs,
        }
        try:
            price = predict_price(model, sample, metadata)
        except FeatureError as exc:
            st.error(str(exc), icon=":material/error:")
            return
        except Exception as exc:  # noqa: BLE001 - surface unexpected model/runtime errors
            log.exception("Prediction failed for %s", upload_name)
            st.error(f"Could not generate a price: {exc}", icon=":material/error:")
            return
        render_result(price, sample, metadata)


if __name__ == "__main__":
    main()