"""Streamlit app: CFD-generated airfoil data to surface-pressure surrogate."""

from __future__ import annotations

import os
import re
import struct
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

MATPLOTLIB_CONFIG_ENV_VAR = "".join(("MPL", "CONFIG", "DIR"))
if MATPLOTLIB_CONFIG_ENV_VAR not in os.environ:
    matplotlib_cache = Path(__file__).parent / ".matplotlib-cache"
    try:
        matplotlib_cache.mkdir(exist_ok=True)
        if not os.access(matplotlib_cache, os.W_OK):
            raise OSError("Matplotlib cache directory is not writable.")
    except OSError:
        matplotlib_cache = Path(tempfile.gettempdir()) / "su2-airfoil-matplotlib"
        matplotlib_cache.mkdir(exist_ok=True)
    os.environ[MATPLOTLIB_CONFIG_ENV_VAR] = str(matplotlib_cache)

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).parent
INPUT_DIR = ROOT / "data" / "input"
DIRECT_DIR = ROOT / "data" / "results" / "direct"
ADJOINT_DIR = ROOT / "data" / "results" / "adjoint"
SWEEP_DIR = ROOT / "data" / "results" / "sweep"

DIRECT_CONFIG = INPUT_DIR / "inv_NACA0012.cfg"
ADJOINT_CONFIG = INPUT_DIR / "inv_NACA0012_adjoint.cfg"
MESH_PATH = INPUT_DIR / "mesh_NACA0012_inv.su2"
DIRECT_HISTORY = DIRECT_DIR / "history_direct.csv"
SURFACE_FLOW = DIRECT_DIR / "surface_flow.csv"
FLOW_VTU = DIRECT_DIR / "flow.vtu"
DIRECT_LOG = DIRECT_DIR / "direct.log"
ADJOINT_HISTORY = ADJOINT_DIR / "history_adjoint.csv"
SURFACE_ADJOINT = ADJOINT_DIR / "surface_adjoint.csv"
ADJOINT_LOG = ADJOINT_DIR / "adjoint.log"
SWEEP_SURFACE = SWEEP_DIR / "surface_pressure_sweep.csv"
SWEEP_SUMMARY = SWEEP_DIR / "sweep_summary.csv"

SU2_QUICK_START_URL = "https://su2code.github.io/docs_v7/Quick-Start/"
SU2_RELEASE_URL = "https://github.com/su2code/SU2/releases/tag/v8.5.0"
DEFAULT_HOLDOUT_CASE = "mach_0p8641_aoa_p0p362"
MODEL_VERSION = "jump-aware-v6-lazy-nav"
JUMP_DETECTION_MIN_SIZE = 0.20
JUMP_METADATA_NEIGHBORS = 8
JUMP_WEIGHT_WIDTH = 0.025
JUMP_WEIGHT_BOOST = 25.0
JUMP_SHOULDER_OFFSET = 0.045
JUMP_SHOULDER_WIDTH = 0.025
JUMP_SHOULDER_WEIGHT_BOOST = 7.0
JUMP_WEIGHT_REFERENCE_SIZE = 0.50
JUMP_UNDERPERFORMANCE_RISK_SCALE = 0.30
RISK_DISTANCE_SCALE = 0.60
RISK_CFD_JUMP_SPAN_SCALE = 1.80
RESPONSE_JUMP_ERROR_RISK_SCALE = 0.25
RESPONSE_JUMP_SIZE_RISK_SCALE = 0.35
RESPONSE_JUMP_RMSE_RISK_SCALE = 0.60
JUMP_AWARE_ERROR_RISK_SCALE = 0.25
JUMP_AWARE_SIZE_RISK_SCALE = 0.62
JUMP_AWARE_RMSE_RISK_SCALE = 0.30
SMOOTH_REGION_DISTANCE_RISK_SCALE = 0.25
SMOOTH_REGION_RMSE_RISK_SCALE = 0.15


class MeshData(NamedTuple):
    """Subset of the SU2 mesh needed for app visualizations."""

    point_count: int
    element_count: int
    marker_counts: dict[str, int]
    points: pd.DataFrame
    airfoil: pd.DataFrame
    farfield: pd.DataFrame


class FlowField(NamedTuple):
    """Unstructured field output parsed from SU2's ParaView VTU file."""

    point_count: int
    cell_count: int
    points: pd.DataFrame
    triangles: np.ndarray
    fields: dict[str, np.ndarray]


class CaseArtifacts(NamedTuple):
    """Single-case SU2 artifacts used as supporting context."""

    direct_config: dict[str, str]
    adjoint_config: dict[str, str]
    mesh: MeshData
    flow_field: FlowField
    direct_history: pd.DataFrame
    surface_flow: pd.DataFrame
    adjoint_history: pd.DataFrame
    surface_adjoint: pd.DataFrame


class SurfaceResponseSurrogate(NamedTuple):
    """Response-surface surrogate over Mach, AoA, and airfoil surface point."""

    cp_grid: pd.DataFrame
    mach_values: tuple[float, ...]
    aoa_values: tuple[float, ...]
    training_rows: int
    training_cases: int
    fit_seconds: float


class SurrogateResult(NamedTuple):
    """A trained surrogate and its held-out-case evaluation."""

    model: SurfaceResponseSurrogate
    holdout_case_id: str
    training_rows: int
    training_cases: int
    fit_seconds: float
    inference_seconds: float
    predictions: pd.DataFrame
    metrics: dict[str, float]


class FullSurrogate(NamedTuple):
    """A surrogate trained on the full CFD sweep for design exploration."""

    model: SurfaceResponseSurrogate
    training_rows: int
    training_cases: int
    fit_seconds: float


class JumpAwareRegressor(NamedTuple):
    """A weighted learned regressor focused on pressure-jump fidelity."""

    model: ExtraTreesRegressor
    jump_catalog: pd.DataFrame
    training_rows: int
    training_cases: int
    fit_seconds: float
    weighted_rows: int
    mean_sample_weight: float
    max_sample_weight: float


class JumpAwareResult(NamedTuple):
    """A jump-aware regressor and its held-out-case evaluation."""

    model: JumpAwareRegressor
    holdout_case_id: str
    training_rows: int
    training_cases: int
    fit_seconds: float
    inference_seconds: float
    predictions: pd.DataFrame
    metrics: dict[str, float]
    jump_zone_metrics: dict[str, float]


class FullJumpAwareSurrogate(NamedTuple):
    """A jump-aware regressor trained on the full CFD sweep."""

    model: JumpAwareRegressor
    training_rows: int
    training_cases: int
    fit_seconds: float


def read_csv(path: Path) -> pd.DataFrame:
    """Load SU2 CSV output and normalize quoted column names."""

    df = pd.read_csv(path)
    df.columns = [column.strip().strip('"') for column in df.columns]
    return df


def parse_config(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE entries from an SU2 config file."""

    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("%", maxsplit=1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        values[key.strip()] = value.strip()
    return values


def _find_line(lines: list[str], prefix: str) -> int:
    """Return the index of the first mesh line starting with a prefix."""

    for index, line in enumerate(lines):
        if line.startswith(prefix):
            return index
    raise ValueError(f"Could not find mesh section: {prefix}")


def parse_mesh(path: Path) -> MeshData:
    """Parse points and boundary markers from a native SU2 mesh."""

    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    element_line = lines[_find_line(lines, "NELEM=")]
    element_count = int(element_line.split("=", maxsplit=1)[1])

    point_start = _find_line(lines, "NPOIN=")
    point_count = int(lines[point_start].split("=", maxsplit=1)[1])
    point_rows = []
    for line in lines[point_start + 1 : point_start + 1 + point_count]:
        parts = line.split()
        point_rows.append(
            {
                "point_id": int(parts[2]),
                "x": float(parts[0]),
                "y": float(parts[1]),
            }
        )
    points = pd.DataFrame(point_rows).set_index("point_id").sort_index()

    marker_counts: dict[str, int] = {}
    marker_points: dict[str, list[int]] = {}
    index = _find_line(lines, "NMARK=") + 1
    while index < len(lines):
        if not lines[index].startswith("MARKER_TAG="):
            index += 1
            continue
        marker_name = lines[index].split("=", maxsplit=1)[1].strip()
        marker_count = int(lines[index + 1].split("=", maxsplit=1)[1])
        marker_counts[marker_name] = marker_count
        ids: list[int] = []
        for marker_line in lines[index + 2 : index + 2 + marker_count]:
            parts = marker_line.split()
            ids.extend(int(value) for value in parts[1:])
        marker_points[marker_name] = list(dict.fromkeys(ids))
        index += 2 + marker_count

    airfoil = points.loc[marker_points["airfoil"]].reset_index()
    farfield = points.loc[marker_points["farfield"]].reset_index()

    return MeshData(
        point_count=point_count,
        element_count=element_count,
        marker_counts=marker_counts,
        points=points.reset_index(),
        airfoil=airfoil,
        farfield=farfield,
    )


def parse_vtu_field(path: Path) -> FlowField:
    """Read the appended-binary VTU fields produced by SU2.

    The file uses VTK XML metadata followed by raw appended binary blocks. This
    parser only handles the subset needed here: points, triangle connectivity,
    and point-data fields.
    """

    raw = path.read_bytes()
    xml_header = raw.split(b"<AppendedData", maxsplit=1)[0].decode(
        "utf-8",
        errors="replace",
    )

    piece_match = re.search(
        r'<Piece NumberOfPoints="(?P<points>\d+)" NumberOfCells="(?P<cells>\d+)">',
        xml_header,
    )
    if piece_match is None:
        raise ValueError("Could not read VTU point/cell counts.")

    point_count = int(piece_match.group("points"))
    cell_count = int(piece_match.group("cells"))

    appended_start = raw.index(b"<AppendedData")
    data_start = raw.index(b"_", appended_start) + 1

    array_pattern = re.compile(
        r'<DataArray type="(?P<type>[^"]+)" Name="(?P<name>[^"]*)"'
        r'\s+NumberOfComponents=\s*"(?P<components>\d+)"'
        r'\s+offset="(?P<offset>\d+)"\s+format="appended"/>'
    )

    dtype_map = {
        "Float32": np.dtype("<f4"),
        "Int32": np.dtype("<i4"),
        "UInt8": np.dtype("u1"),
    }

    arrays: dict[str, np.ndarray] = {}
    for match in array_pattern.finditer(xml_header):
        vtk_type = match.group("type")
        dtype = dtype_map[vtk_type]
        name = match.group("name") or "Points"
        components = int(match.group("components"))
        offset = int(match.group("offset"))

        block_start = data_start + offset
        byte_count = struct.unpack_from("<Q", raw, block_start)[0]
        array_start = block_start + 8
        array_end = array_start + byte_count
        data = np.frombuffer(raw[array_start:array_end], dtype=dtype).copy()
        if components > 1:
            data = data.reshape((-1, components))
        arrays[name] = data

    points_array = arrays["Points"]
    points = pd.DataFrame(
        {
            "x": points_array[:, 0],
            "y": points_array[:, 1],
            "z": points_array[:, 2],
        }
    )

    connectivity = arrays["connectivity"]
    offsets = arrays["offsets"]
    types = arrays["types"]
    triangles = []
    start = 0
    for offset, cell_type in zip(offsets, types, strict=True):
        stop = int(offset)
        cell = connectivity[start:stop]
        if int(cell_type) == 5 and len(cell) == 3:
            triangles.append(cell)
        start = stop

    fields = {
        name: data
        for name, data in arrays.items()
        if name
        not in {
            "Points",
            "connectivity",
            "offsets",
            "types",
        }
    }

    return FlowField(
        point_count=point_count,
        cell_count=cell_count,
        points=points,
        triangles=np.asarray(triangles, dtype=np.int32),
        fields=fields,
    )


@st.cache_resource(show_spinner=False)
def load_case_artifacts() -> CaseArtifacts:
    """Load local SU2 artifacts for the reference case."""

    return CaseArtifacts(
        direct_config=parse_config(DIRECT_CONFIG),
        adjoint_config=parse_config(ADJOINT_CONFIG),
        mesh=parse_mesh(MESH_PATH),
        flow_field=parse_vtu_field(FLOW_VTU),
        direct_history=read_csv(DIRECT_HISTORY),
        surface_flow=read_csv(SURFACE_FLOW),
        adjoint_history=read_csv(ADJOINT_HISTORY),
        surface_adjoint=read_csv(SURFACE_ADJOINT),
    )


@st.cache_data(show_spinner=False)
def load_sweep_data(data_version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the dense CFD sweep used to train and validate the surrogate."""

    del data_version
    surface = read_csv(SWEEP_SURFACE)
    summary = read_csv(SWEEP_SUMMARY)

    surface["case_label"] = (
        "Mach "
        + surface["mach"].map("{:.3f}".format)
        + ", AoA "
        + surface["aoa_deg"].map("{:+.2f} deg".format)
    )
    summary["case_label"] = (
        "Mach "
        + summary["mach"].map("{:.3f}".format)
        + ", AoA "
        + summary["aoa_deg"].map("{:+.2f} deg".format)
    )

    return surface, summary


def data_version_token() -> str:
    """Return a cheap cache-busting token for generated sweep artifacts."""

    paths = [SWEEP_SURFACE, SWEEP_SUMMARY]
    return "|".join(
        f"{path.stat().st_mtime_ns}:{path.stat().st_size}" for path in paths
    )


def latest_value(df: pd.DataFrame, column: str) -> float | None:
    """Return the last finite value for a column, if present."""

    if column not in df:
        return None
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def value_or_dash(value: float | None, digits: int = 4) -> str:
    """Format optional floats for metric cards."""

    if value is None:
        return "-"
    return f"{value:.{digits}g}"


def train_response_surrogate(surface: pd.DataFrame) -> SurfaceResponseSurrogate:
    """Build a local response-surface surrogate from CFD pressure curves.

    This is a deliberately conservative surrogate for a fixed geometry: for each
    airfoil surface point, it stores Cp over the generated Mach/AoA grid and uses
    local bilinear interpolation for new operating conditions.
    """

    started_at = time.perf_counter()
    cp_grid = surface.pivot_table(
        index="PointID",
        columns=["mach", "aoa_deg"],
        values="cp",
        aggfunc="first",
    ).sort_index(axis=0)
    cp_grid = cp_grid.sort_index(axis=1)
    fit_seconds = time.perf_counter() - started_at

    mach_values = tuple(float(value) for value in sorted(surface["mach"].unique()))
    aoa_values = tuple(float(value) for value in sorted(surface["aoa_deg"].unique()))

    return SurfaceResponseSurrogate(
        cp_grid=cp_grid,
        mach_values=mach_values,
        aoa_values=aoa_values,
        training_rows=len(surface),
        training_cases=surface["case_id"].nunique(),
        fit_seconds=fit_seconds,
    )


def _axis_weights(
    values: tuple[float, ...],
    target: float,
    force_span: bool,
) -> list[tuple[float, float]]:
    """Return interpolation or extrapolation weights along one axis."""

    axis = np.asarray(values, dtype=float)
    exact_index = np.where(np.isclose(axis, target, atol=1e-9))[0]
    if len(exact_index) and not force_span:
        return [(float(axis[int(exact_index[0])]), 1.0)]

    if target <= axis[0]:
        low, high = float(axis[0]), float(axis[1])
    elif target >= axis[-1]:
        low, high = float(axis[-2]), float(axis[-1])
    else:
        lower_candidates = axis[axis < target]
        upper_candidates = axis[axis > target]
        if len(lower_candidates) == 0:
            low, high = float(axis[0]), float(axis[1])
        elif len(upper_candidates) == 0:
            low, high = float(axis[-2]), float(axis[-1])
        else:
            low = float(lower_candidates.max())
            high = float(upper_candidates.min())

    high_weight = (target - low) / (high - low)
    return [(low, 1.0 - high_weight), (high, high_weight)]


def _has_exact_column(
    surrogate: SurfaceResponseSurrogate,
    mach: float,
    aoa_deg: float,
) -> bool:
    """Return whether the response surface contains this exact generated case."""

    columns = surrogate.cp_grid.columns
    for column_mach, column_aoa in columns:
        if np.isclose(column_mach, mach, atol=1e-9) and np.isclose(
            column_aoa,
            aoa_deg,
            atol=1e-9,
        ):
            return True
    return False


def _column_key(
    surrogate: SurfaceResponseSurrogate,
    mach: float,
    aoa_deg: float,
) -> tuple[float, float]:
    """Return the exact MultiIndex key for one CFD grid column."""

    for column_mach, column_aoa in surrogate.cp_grid.columns:
        if np.isclose(column_mach, mach, atol=1e-9) and np.isclose(
            column_aoa,
            aoa_deg,
            atol=1e-9,
        ):
            return float(column_mach), float(column_aoa)
    raise KeyError(f"No CFD grid column for Mach {mach} and AoA {aoa_deg}.")


def _optional_column_key(
    surrogate: SurfaceResponseSurrogate,
    mach: float,
    aoa_deg: float,
) -> tuple[float, float] | None:
    """Return a CFD grid column key when it exists."""

    try:
        return _column_key(surrogate, mach, aoa_deg)
    except KeyError:
        return None


def _nearest_column_weights(
    surrogate: SurfaceResponseSurrogate,
    mach: float,
    aoa_deg: float,
    limit: int = 6,
) -> list[tuple[tuple[float, float], float]]:
    """Return inverse-distance weights for nearby available CFD columns."""

    mach_span = max(surrogate.mach_values) - min(surrogate.mach_values)
    aoa_span = max(surrogate.aoa_values) - min(surrogate.aoa_values)
    mach_scale = mach_span if mach_span > 0.0 else 1.0
    aoa_scale = aoa_span if aoa_span > 0.0 else 1.0

    distances = []
    for column_mach, column_aoa in surrogate.cp_grid.columns:
        distance = np.sqrt(
            ((float(column_mach) - mach) / mach_scale) ** 2
            + ((float(column_aoa) - aoa_deg) / aoa_scale) ** 2
        )
        distances.append(((float(column_mach), float(column_aoa)), float(distance)))

    nearest = sorted(distances, key=lambda item: item[1])[:limit]
    raw_weights = np.asarray([1.0 / max(distance, 1e-9) for _key, distance in nearest])
    normalized_weights = raw_weights / raw_weights.sum()
    return [
        (key, float(weight))
        for (key, _distance), weight in zip(nearest, normalized_weights, strict=True)
    ]


def response_surface_predict(
    surrogate: SurfaceResponseSurrogate,
    point_ids: pd.Series,
    mach: float,
    aoa_deg: float,
) -> np.ndarray:
    """Predict Cp for the requested surface points and operating condition."""

    has_exact_case = _has_exact_column(surrogate, mach, aoa_deg)
    if has_exact_case:
        exact_key = _column_key(surrogate, mach, aoa_deg)
        return surrogate.cp_grid.loc[point_ids, exact_key].to_numpy(dtype=float)

    mach_is_exact = any(
        np.isclose(value, mach, atol=1e-9) for value in surrogate.mach_values
    )
    aoa_is_exact = any(
        np.isclose(value, aoa_deg, atol=1e-9) for value in surrogate.aoa_values
    )
    force_span = mach_is_exact and aoa_is_exact
    mach_weights = _axis_weights(
        surrogate.mach_values,
        mach,
        force_span=force_span,
    )
    aoa_weights = _axis_weights(
        surrogate.aoa_values,
        aoa_deg,
        force_span=force_span,
    )

    prediction = np.zeros(len(point_ids), dtype=float)
    weighted_columns = []
    for weighted_mach, mach_weight in mach_weights:
        for weighted_aoa, aoa_weight in aoa_weights:
            key = _optional_column_key(surrogate, weighted_mach, weighted_aoa)
            if key is None:
                weighted_columns = []
                break
            weighted_columns.append((key, mach_weight * aoa_weight))
        if not weighted_columns:
            break

    if not weighted_columns:
        weighted_columns = _nearest_column_weights(surrogate, mach, aoa_deg)

    for key, weight in weighted_columns:
        prediction += weight * surrogate.cp_grid.loc[point_ids, key].to_numpy(
            dtype=float
        )
    return prediction


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics for pressure coefficient prediction."""

    y_true_array = y_true.to_numpy(dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    residual = y_true_array - y_pred_array
    total = y_true_array - y_true_array.mean()
    total_sum_squares = float(np.sum(total**2))
    residual_sum_squares = float(np.sum(residual**2))
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "R2": float(1.0 - residual_sum_squares / total_sum_squares),
    }


@st.cache_resource(show_spinner="Training holdout surrogate...")
def train_holdout_surrogate(
    holdout_case_id: str,
    data_version: str,
    model_version: str,
) -> SurrogateResult:
    """Train on all but one CFD case and evaluate on one unseen Mach/AoA case."""

    del model_version
    surface, _summary = load_sweep_data(data_version)
    train_df = surface[surface["case_id"] != holdout_case_id]
    holdout_df = surface[surface["case_id"] == holdout_case_id].copy()

    model = train_response_surrogate(train_df)

    started_at = time.perf_counter()
    predicted_cp = response_surface_predict(
        surrogate=model,
        point_ids=holdout_df["PointID"],
        mach=float(holdout_df["mach"].iloc[0]),
        aoa_deg=float(holdout_df["aoa_deg"].iloc[0]),
    )
    inference_seconds = time.perf_counter() - started_at

    holdout_df["cp_pred"] = predicted_cp
    holdout_df["error"] = holdout_df["cp_pred"] - holdout_df["cp"]
    holdout_df["abs_error"] = holdout_df["error"].abs()

    return SurrogateResult(
        model=model,
        holdout_case_id=holdout_case_id,
        training_rows=model.training_rows,
        training_cases=model.training_cases,
        fit_seconds=model.fit_seconds,
        inference_seconds=inference_seconds,
        predictions=holdout_df.sort_values(["surface", "x"]),
        metrics=compute_metrics(holdout_df["cp"], predicted_cp),
    )


@st.cache_resource(show_spinner="Training full sweep surrogate...")
def train_full_surrogate(data_version: str, model_version: str) -> FullSurrogate:
    """Train on all generated CFD cases for interactive design exploration."""

    del model_version
    surface, _summary = load_sweep_data(data_version)
    model = train_response_surrogate(surface)
    return FullSurrogate(
        model=model,
        training_rows=model.training_rows,
        training_cases=model.training_cases,
        fit_seconds=model.fit_seconds,
    )


def estimate_jump_metadata(
    surface: pd.DataFrame,
    jump_catalog: pd.DataFrame,
    use_case_jumps: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate local pressure-jump metadata for regressor feature engineering."""

    strong_jumps = jump_catalog[jump_catalog["jump_size"] > JUMP_DETECTION_MIN_SIZE]

    columns = ["case_id", "surface", "jump_x", "jump_size"]
    if use_case_jumps and "case_id" in surface:
        merged = surface.merge(
            strong_jumps[columns],
            on=["case_id", "surface"],
            how="left",
        )
        return (
            merged["jump_x"].fillna(0.5).to_numpy(dtype=float),
            merged["jump_size"].fillna(0.0).to_numpy(dtype=float),
        )

    mach_span = float(jump_catalog["mach"].max() - jump_catalog["mach"].min())
    aoa_span = float(jump_catalog["aoa_deg"].max() - jump_catalog["aoa_deg"].min())
    mach_scale = mach_span if mach_span > 0.0 else 1.0
    aoa_scale = aoa_span if aoa_span > 0.0 else 1.0
    lookup: dict[tuple[float, float, str], tuple[float, float]] = {}
    for row in (
        surface[["mach", "aoa_deg", "surface"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        side = jump_catalog[jump_catalog["surface"] == row.surface].copy()
        distance = np.sqrt(
            ((side["mach"] - row.mach) / mach_scale) ** 2
            + ((side["aoa_deg"] - row.aoa_deg) / aoa_scale) ** 2
        )
        nearest = side.assign(distance=distance).nsmallest(
            JUMP_METADATA_NEIGHBORS,
            "distance",
        )
        weights = 1.0 / np.maximum(nearest["distance"].to_numpy(dtype=float), 1e-8)
        weights *= 1.0 + nearest["jump_size"].to_numpy(dtype=float)
        weights /= weights.sum()

        local_jump_size = float(
            np.sum(nearest["jump_size"].to_numpy(dtype=float) * weights)
        )
        strong_nearest = nearest[nearest["jump_size"] > JUMP_DETECTION_MIN_SIZE]
        if strong_nearest.empty or local_jump_size <= JUMP_DETECTION_MIN_SIZE:
            lookup[(float(row.mach), float(row.aoa_deg), str(row.surface))] = (
                0.5,
                0.0,
            )
            continue

        weights = 1.0 / np.maximum(
            strong_nearest["distance"].to_numpy(dtype=float),
            1e-8,
        )
        weights *= 1.0 + strong_nearest["jump_size"].to_numpy(dtype=float)
        weights /= weights.sum()
        lookup[(float(row.mach), float(row.aoa_deg), str(row.surface))] = (
            float(np.sum(strong_nearest["jump_x"].to_numpy(dtype=float) * weights)),
            local_jump_size,
        )

    keys = zip(
        surface["mach"].astype(float),
        surface["aoa_deg"].astype(float),
        surface["surface"].astype(str),
        strict=True,
    )
    estimates = [lookup[(mach, aoa, surface_name)] for mach, aoa, surface_name in keys]
    jump_x, jump_size = zip(*estimates, strict=True)
    return np.asarray(jump_x, dtype=float), np.asarray(jump_size, dtype=float)


def regressor_features(
    surface: pd.DataFrame,
    jump_catalog: pd.DataFrame,
    use_case_jumps: bool,
) -> pd.DataFrame:
    """Build numeric features for the learned pressure-coefficient regressor."""

    mach = surface["mach"].astype(float)
    aoa = surface["aoa_deg"].astype(float)
    x = surface["x"].astype(float)
    y = surface["y"].astype(float)
    surface_upper = (surface["surface"] == "upper").astype(float)
    jump_x, jump_size = estimate_jump_metadata(
        surface,
        jump_catalog,
        use_case_jumps=use_case_jumps,
    )
    x_minus_jump = x.to_numpy(dtype=float) - jump_x

    return pd.DataFrame(
        {
            "mach": mach,
            "aoa_deg": aoa,
            "x": x,
            "y": y,
            "surface_upper": surface_upper,
            "mach_x": mach * x,
            "aoa_x": aoa * x,
            "mach_aoa": mach * aoa,
            "x2": x**2,
            "x3": x**3,
            "jump_x_est": jump_x,
            "jump_size_est": jump_size,
            "x_minus_jump": x_minus_jump,
            "abs_x_minus_jump": np.abs(x_minus_jump),
            "after_jump": (x_minus_jump >= 0.0).astype(float),
            "jump_zone_kernel": np.exp(-0.5 * (x_minus_jump / 0.055) ** 2),
            "pre_shoulder_kernel": np.exp(
                -0.5
                * ((x_minus_jump + JUMP_SHOULDER_OFFSET) / JUMP_SHOULDER_WIDTH) ** 2
            ),
            "post_shoulder_kernel": np.exp(
                -0.5
                * ((x_minus_jump - JUMP_SHOULDER_OFFSET) / JUMP_SHOULDER_WIDTH) ** 2
            ),
        }
    )


def jump_sample_weights(
    surface: pd.DataFrame,
    jump_catalog: pd.DataFrame,
) -> np.ndarray:
    """Assign higher training weight near detected pressure jumps.

    The model still sees the full pressure curve. The weighting only changes the
    optimization target: mistakes near strong pressure jumps matter more than
    equally sized mistakes in smooth regions.
    """

    jump_x, jump_size = estimate_jump_metadata(
        surface,
        jump_catalog[jump_catalog["jump_size"] > JUMP_DETECTION_MIN_SIZE],
        use_case_jumps=True,
    )
    x = surface["x"].to_numpy(dtype=float)
    distance = np.abs(x - jump_x)
    strength = np.clip(jump_size / JUMP_WEIGHT_REFERENCE_SIZE, 0.0, 1.5)
    center_proximity = np.exp(-0.5 * (distance / JUMP_WEIGHT_WIDTH) ** 2)
    pre_shoulder = np.exp(
        -0.5 * ((x - jump_x + JUMP_SHOULDER_OFFSET) / JUMP_SHOULDER_WIDTH) ** 2
    )
    post_shoulder = np.exp(
        -0.5 * ((x - jump_x - JUMP_SHOULDER_OFFSET) / JUMP_SHOULDER_WIDTH) ** 2
    )
    shoulder_proximity = np.maximum(pre_shoulder, post_shoulder)
    weights = (
        1.0
        + JUMP_WEIGHT_BOOST * center_proximity * strength
        + JUMP_SHOULDER_WEIGHT_BOOST * shoulder_proximity * strength
    )
    return np.asarray(weights, dtype=float)


def train_jump_aware_regressor(
    surface: pd.DataFrame,
    jump_catalog: pd.DataFrame,
) -> JumpAwareRegressor:
    """Train a weighted regressor for Cp over Mach, AoA, and surface location."""

    sample_weights = jump_sample_weights(surface, jump_catalog)
    model = ExtraTreesRegressor(
        n_estimators=120,
        max_depth=28,
        min_samples_leaf=2,
        random_state=17,
        n_jobs=1,
    )

    started_at = time.perf_counter()
    model.fit(
        regressor_features(
            surface,
            jump_catalog,
            use_case_jumps=True,
        ),
        surface["cp"],
        sample_weight=sample_weights,
    )
    fit_seconds = time.perf_counter() - started_at

    return JumpAwareRegressor(
        model=model,
        jump_catalog=jump_catalog.copy(),
        training_rows=len(surface),
        training_cases=surface["case_id"].nunique(),
        fit_seconds=fit_seconds,
        weighted_rows=int(np.count_nonzero(sample_weights > 1.10)),
        mean_sample_weight=float(sample_weights.mean()),
        max_sample_weight=float(sample_weights.max()),
    )


def compute_jump_zone_metrics(
    predictions: pd.DataFrame,
    width: float = 0.06,
) -> dict[str, float]:
    """Compute metrics only near the SU2 pressure-jump locations."""

    jumps = pressure_jump_table(predictions, "cp", "SU2")
    if jumps.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Rows": 0.0}

    mask = pd.Series(False, index=predictions.index)
    for row in jumps.itertuples(index=False):
        local_mask = (predictions["surface"] == row.surface) & (
            (predictions["x"] - float(row.jump_x)).abs() <= width
        )
        mask = mask | local_mask

    jump_zone = predictions[mask]
    if jump_zone.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Rows": 0.0}

    metrics = compute_metrics(jump_zone["cp"], jump_zone["cp_pred"].to_numpy())
    metrics["Rows"] = float(len(jump_zone))
    return metrics


@st.cache_resource(show_spinner="Training jump-aware holdout regressor...")
def train_jump_aware_holdout_surrogate(
    holdout_case_id: str,
    data_version: str,
    model_version: str,
) -> JumpAwareResult:
    """Train the weighted regressor without the selected CFD case."""

    del model_version
    surface, _summary = load_sweep_data(data_version)
    train_df = surface[surface["case_id"] != holdout_case_id]
    holdout_df = surface[surface["case_id"] == holdout_case_id].copy()
    train_jumps = build_jump_catalog(train_df)

    model = train_jump_aware_regressor(train_df, train_jumps)

    started_at = time.perf_counter()
    holdout_df["cp_pred"] = model.model.predict(
        regressor_features(
            holdout_df,
            model.jump_catalog,
            use_case_jumps=False,
        )
    )
    inference_seconds = time.perf_counter() - started_at
    holdout_df["error"] = holdout_df["cp_pred"] - holdout_df["cp"]
    holdout_df["abs_error"] = holdout_df["error"].abs()

    predictions = holdout_df.sort_values(["surface", "x"])
    return JumpAwareResult(
        model=model,
        holdout_case_id=holdout_case_id,
        training_rows=model.training_rows,
        training_cases=model.training_cases,
        fit_seconds=model.fit_seconds,
        inference_seconds=inference_seconds,
        predictions=predictions,
        metrics=compute_metrics(holdout_df["cp"], holdout_df["cp_pred"].to_numpy()),
        jump_zone_metrics=compute_jump_zone_metrics(predictions),
    )


@st.cache_resource(show_spinner="Training full jump-aware regressor...")
def train_full_jump_aware_surrogate(
    data_version: str,
    model_version: str,
) -> FullJumpAwareSurrogate:
    """Train the weighted regressor on every generated CFD case."""

    del model_version
    surface, _summary = load_sweep_data(data_version)
    jump_catalog = build_jump_catalog(surface)
    model = train_jump_aware_regressor(surface, jump_catalog)
    return FullJumpAwareSurrogate(
        model=model,
        training_rows=model.training_rows,
        training_cases=model.training_cases,
        fit_seconds=model.fit_seconds,
    )


def predict_surface_curve(
    model: SurfaceResponseSurrogate,
    surface_template: pd.DataFrame,
    mach: float,
    aoa_deg: float,
) -> tuple[pd.DataFrame, float]:
    """Predict a full airfoil surface-pressure curve for one condition."""

    prediction_frame = surface_template.copy()
    prediction_frame["mach"] = mach
    prediction_frame["aoa_deg"] = aoa_deg

    started_at = time.perf_counter()
    prediction_frame["cp_pred"] = response_surface_predict(
        surrogate=model,
        point_ids=prediction_frame["PointID"],
        mach=mach,
        aoa_deg=aoa_deg,
    )
    inference_seconds = time.perf_counter() - started_at

    return prediction_frame.sort_values(["surface", "x"]), inference_seconds


def predict_jump_aware_curve(
    model: JumpAwareRegressor,
    surface_template: pd.DataFrame,
    mach: float,
    aoa_deg: float,
) -> tuple[pd.DataFrame, float]:
    """Predict a pressure curve using the weighted learned regressor."""

    prediction_frame = surface_template.copy()
    prediction_frame["mach"] = mach
    prediction_frame["aoa_deg"] = aoa_deg

    started_at = time.perf_counter()
    prediction_frame["cp_pred"] = model.model.predict(
        regressor_features(
            prediction_frame,
            model.jump_catalog,
            use_case_jumps=False,
        )
    )
    inference_seconds = time.perf_counter() - started_at

    return prediction_frame.sort_values(["surface", "x"]), inference_seconds


def format_case(summary_by_case: pd.DataFrame, case_id: str) -> str:
    """Return a readable label for a sweep case id."""

    row = summary_by_case.loc[case_id]
    return f"Mach {row['mach']:.3f}, AoA {row['aoa_deg']:+.2f} deg"


def nearest_design_cases(
    summary: pd.DataFrame,
    mach: float,
    aoa_deg: float,
    limit: int = 4,
    excluded_case_id: str | None = None,
) -> pd.DataFrame:
    """Find nearby CFD cases in normalized Mach/AoA space."""

    candidates = summary.copy()
    if excluded_case_id is not None:
        candidates = candidates[candidates["case_id"] != excluded_case_id]

    mach_span = float(summary["mach"].max() - summary["mach"].min())
    aoa_span = float(summary["aoa_deg"].max() - summary["aoa_deg"].min())
    mach_scale = mach_span if mach_span > 0.0 else 1.0
    aoa_scale = aoa_span if aoa_span > 0.0 else 1.0

    candidates["normalized_distance"] = np.sqrt(
        ((candidates["mach"] - mach) / mach_scale) ** 2
        + ((candidates["aoa_deg"] - aoa_deg) / aoa_scale) ** 2
    )
    return candidates.nsmallest(limit, "normalized_distance")


def generated_value_options(summary: pd.DataFrame) -> tuple[list[float], list[float]]:
    """Return exact Mach and AoA values available in the generated sweep."""

    mach_values = sorted(float(value) for value in summary["mach"].unique())
    aoa_values = sorted(float(value) for value in summary["aoa_deg"].unique())
    return mach_values, aoa_values


def available_aoa_options(summary: pd.DataFrame, mach: float) -> list[float]:
    """Return generated AoA values available for one Mach value."""

    matching = summary[np.isclose(summary["mach"], mach, atol=1e-9)]
    return sorted(float(value) for value in matching["aoa_deg"].unique())


def exact_case_id(summary: pd.DataFrame, mach: float, aoa_deg: float) -> str | None:
    """Return the CFD case id when a requested design is already in the sweep."""

    exact = summary[
        np.isclose(summary["mach"], mach, atol=1e-9)
        & np.isclose(summary["aoa_deg"], aoa_deg, atol=1e-9)
    ]
    if exact.empty:
        return None
    return str(exact.iloc[0]["case_id"])


def is_boundary_case(summary: pd.DataFrame, case_id: str) -> bool:
    """Return whether a case lies on the generated Mach/AoA envelope boundary."""

    row = summary.loc[summary["case_id"] == case_id].iloc[0]
    return bool(
        np.isclose(row["mach"], summary["mach"].min(), atol=1e-9)
        or np.isclose(row["mach"], summary["mach"].max(), atol=1e-9)
        or np.isclose(row["aoa_deg"], summary["aoa_deg"].min(), atol=1e-9)
        or np.isclose(row["aoa_deg"], summary["aoa_deg"].max(), atol=1e-9)
    )


def interpolation_status(
    summary: pd.DataFrame,
    mach: float,
    aoa_deg: float,
) -> tuple[str, str]:
    """Classify whether a requested design is inside the sweep envelope."""

    mach_min = float(summary["mach"].min())
    mach_max = float(summary["mach"].max())
    aoa_min = float(summary["aoa_deg"].min())
    aoa_max = float(summary["aoa_deg"].max())
    inside_mach = mach_min <= mach <= mach_max
    inside_aoa = aoa_min <= aoa_deg <= aoa_max

    if inside_mach and inside_aoa:
        return (
            "Interpolation",
            "The requested condition sits inside the generated Mach/AoA envelope.",
        )
    return (
        "Extrapolation",
        "The requested condition is outside the generated CFD envelope; treat the "
        "curve as a prompt for more simulation, not as a trusted prediction.",
    )


def solver_time_for_case(summary: pd.DataFrame, case_id: str) -> float:
    """Return the measured local SU2 wall time for one sweep case."""

    row = summary.loc[summary["case_id"] == case_id].iloc[0]
    return float(row["runtime_seconds"])


def safe_speedup(solver_seconds: float, surrogate_seconds: float) -> float:
    """Compute speedup while avoiding division by zero."""

    return solver_seconds / max(surrogate_seconds, 1e-9)


def pressure_jump_table(
    curve: pd.DataFrame,
    cp_column: str,
    source: str,
) -> pd.DataFrame:
    """Detect the dominant pressure jump on each airfoil side."""

    rows = []
    for surface_name, side in curve.groupby("surface"):
        ordered = side.sort_values("x")
        x = ordered["x"].to_numpy(dtype=float)
        cp = ordered[cp_column].to_numpy(dtype=float)
        if len(x) < 3:
            continue

        dx = np.diff(x)
        dcp = np.diff(cp)
        slope = np.abs(dcp / np.maximum(np.abs(dx), 1e-9))
        interior = (x[:-1] > 0.10) & (x[:-1] < 0.90)
        if not interior.any():
            continue

        interior_indices = np.where(interior)[0]
        jump_index = int(interior_indices[np.argmax(slope[interior_indices])])
        rows.append(
            {
                "surface": surface_name,
                "source": source,
                "jump_x": float((x[jump_index] + x[jump_index + 1]) / 2.0),
                "jump_size": float(abs(dcp[jump_index])),
                "jump_slope": float(slope[jump_index]),
            }
        )
    return pd.DataFrame(rows)


def jump_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare detected pressure-jump locations between SU2 and surrogate."""

    su2_jumps = pressure_jump_table(predictions, "cp", "SU2").set_index("surface")
    predicted_jumps = pressure_jump_table(
        predictions,
        "cp_pred",
        "Surrogate",
    ).set_index("surface")
    compared = su2_jumps.join(
        predicted_jumps,
        lsuffix="_su2",
        rsuffix="_surrogate",
    )
    compared["jump_x_error"] = compared["jump_x_surrogate"] - compared["jump_x_su2"]
    compared["jump_size_error"] = (
        compared["jump_size_surrogate"] - compared["jump_size_su2"]
    )
    return compared.reset_index()


def build_jump_catalog(surface: pd.DataFrame) -> pd.DataFrame:
    """Build pressure-jump metadata for every generated CFD case."""

    rows = []
    metadata_columns = ["case_id", "mach", "aoa_deg"]
    for _case_id, case_df in surface.groupby("case_id"):
        metadata = case_df.iloc[0][metadata_columns].to_dict()
        jumps = pressure_jump_table(case_df, "cp", "SU2")
        for row in jumps.to_dict("records"):
            rows.append({**metadata, **row})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_jump_catalog(data_version: str) -> pd.DataFrame:
    """Load cached pressure-jump metadata for the generated sweep."""

    surface, _summary = load_sweep_data(data_version)
    return build_jump_catalog(surface)


def render_jump_map(jump_catalog: pd.DataFrame) -> None:
    """Render where the dominant pressure jump appears across the sweep."""

    plotted = jump_catalog[jump_catalog["jump_size"] > JUMP_DETECTION_MIN_SIZE].copy()
    if plotted.empty:
        st.info("No dominant pressure jumps detected in the generated sweep.")
        return

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Lower surface", "Upper surface"),
        horizontal_spacing=0.08,
    )
    for column_index, surface_name in enumerate(["lower", "upper"], start=1):
        side = plotted[plotted["surface"] == surface_name]
        heatmap = side.pivot_table(
            index="aoa_deg",
            columns="mach",
            values="jump_x",
            aggfunc="mean",
        ).sort_index()
        hover_text = side.pivot_table(
            index="aoa_deg",
            columns="mach",
            values="jump_size",
            aggfunc="mean",
        ).reindex_like(heatmap)
        fig.add_trace(
            go.Heatmap(
                x=heatmap.columns,
                y=heatmap.index,
                z=heatmap.values,
                customdata=hover_text.values,
                coloraxis="coloraxis",
                hovertemplate=(
                    "Mach=%{x:.4f}<br>"
                    "AoA=%{y:.3f} deg<br>"
                    "jump x=%{z:.3f}<br>"
                    "Cp jump=%{customdata:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=column_index,
        )

    fig.update_xaxes(title_text="Mach number")
    fig.update_yaxes(title_text="Angle of attack (deg)", row=1, col=1)
    fig.update_yaxes(title_text="", row=1, col=2)
    fig.update_layout(
        title="Detected pressure-jump location across generated CFD cases",
        height=430,
        margin={"r": 120},
        coloraxis={
            "colorscale": "Turbo",
            "colorbar": {
                "title": {"text": "jump x / chord"},
                "x": 1.03,
            },
        },
    )
    st.plotly_chart(fig, width="stretch")


def apply_layout_styles() -> None:
    """Apply compact Streamlit styling for presentation-sized browser windows."""

    st.markdown(
        """
<style>
  .block-container {
    max-width: 1280px;
    padding-top: 1.35rem;
    padding-bottom: 2.0rem;
  }
  h1 {
    font-size: 2.25rem !important;
    line-height: 1.12 !important;
    margin-bottom: 0.25rem !important;
  }
  h2, h3 {
    margin-top: 0.75rem !important;
    margin-bottom: 0.35rem !important;
  }
  div[data-testid="stMetric"] {
    min-height: 3.55rem;
  }
  div[data-testid="stMetricLabel"] p {
    font-size: 0.82rem;
  }
  div[data-testid="stMetricValue"] {
    font-size: 1.55rem;
  }
  .stTabs [data-baseweb="tab-list"] {
    gap: 0.45rem;
  }
  .stTabs [data-baseweb="tab"] {
    height: 2.35rem;
    padding: 0 0.75rem;
  }
  div[data-testid="stAlert"] {
    padding: 0.65rem 0.85rem;
  }
  div[data-testid="stCaptionContainer"] {
    margin-top: 0.15rem;
  }
  .compact-status {
    border-radius: 0.45rem;
    padding: 0.45rem 0.65rem;
    margin: 0.25rem 0 0.65rem 0;
    font-size: 0.9rem;
  }
  .compact-status.ok {
    background: rgba(46, 204, 113, 0.16);
    color: #58e081;
  }
  .compact-status.warn {
    background: rgba(241, 196, 15, 0.16);
    color: #f4d35e;
  }
  .compact-strip {
    border: 1px solid rgba(250, 250, 250, 0.12);
    border-radius: 0.45rem;
    padding: 0.55rem 0.7rem;
    margin: 0.25rem 0 0.75rem 0;
  }
  .compact-strip strong {
    color: #fafafa;
  }
  .compact-strip span {
    color: #a6a8ad;
    margin-right: 1.15rem;
    white-space: nowrap;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(label: str, text: str, is_ok: bool) -> None:
    """Render a compact interpolation/extrapolation status line."""

    class_name = "ok" if is_ok else "warn"
    st.markdown(
        f"""
<div class="compact-status {class_name}">
  <strong>{label}</strong>: {text}
</div>
        """,
        unsafe_allow_html=True,
    )


def render_compact_strip(items: list[tuple[str, str]]) -> None:
    """Render compact key-value facts without Streamlit metric vertical height."""

    html_items = "\n".join(
        f"<span>{label}: <strong>{value}</strong></span>" for label, value in items
    )
    st.markdown(
        f"""
<div class="compact-strip">
  {html_items}
</div>
        """,
        unsafe_allow_html=True,
    )


def nearest_jump_context(
    jump_catalog: pd.DataFrame,
    nearest: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize pressure-jump movement across nearby supporting CFD cases."""

    nearest_jumps = jump_catalog[
        jump_catalog["case_id"].isin(nearest["case_id"])
        & (jump_catalog["jump_size"] > JUMP_DETECTION_MIN_SIZE)
    ]
    if nearest_jumps.empty:
        return pd.DataFrame()

    jump_context = (
        nearest_jumps.groupby("surface")
        .agg(
            jump_x_min=("jump_x", "min"),
            jump_x_max=("jump_x", "max"),
            max_jump=("jump_size", "max"),
        )
        .reset_index()
    )
    jump_context["jump_x_span"] = (
        jump_context["jump_x_max"] - jump_context["jump_x_min"]
    )
    return jump_context


def aligned_prediction_frame(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
) -> pd.DataFrame:
    """Attach predicted Cp values to the matching CFD truth rows."""

    return truth.drop(columns=["cp_pred"], errors="ignore").merge(
        prediction[["PointID", "cp_pred"]],
        on="PointID",
        how="left",
    )


def max_jump_error_against_truth(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
) -> float:
    """Return max detected jump-location error against a supporting CFD case."""

    compared = jump_comparison(aligned_prediction_frame(truth, prediction))
    compared = compared[compared["jump_size_su2"] > JUMP_DETECTION_MIN_SIZE]
    if compared.empty:
        return 0.0
    return float(compared["jump_x_error"].abs().max())


def max_jump_size_error_against_truth(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
) -> float:
    """Return max detected jump-size error against a supporting CFD case."""

    compared = jump_comparison(aligned_prediction_frame(truth, prediction))
    compared = compared[compared["jump_size_su2"] > JUMP_DETECTION_MIN_SIZE]
    if compared.empty:
        return 0.0
    return float(compared["jump_size_error"].abs().max())


def full_curve_rmse_against_truth(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
) -> float:
    """Return whole-curve RMSE against a supporting CFD case."""

    comparison = aligned_prediction_frame(truth, prediction)
    residual = comparison["cp_pred"].to_numpy(dtype=float) - comparison["cp"].to_numpy(
        dtype=float
    )
    return float(np.sqrt(np.mean(residual**2)))


def jump_zone_rmse_against_truth(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
) -> float:
    """Return jump-zone RMSE against a supporting CFD case."""

    metrics = compute_jump_zone_metrics(aligned_prediction_frame(truth, prediction))
    return float(metrics["RMSE"])


def max_model_jump_disagreement(
    response_prediction: pd.DataFrame,
    jump_prediction: pd.DataFrame,
) -> float:
    """Return max jump-location disagreement between the two surrogate models."""

    response_jumps = pressure_jump_table(
        response_prediction,
        "cp_pred",
        "Response surface",
    ).set_index("surface")
    jump_jumps = pressure_jump_table(
        jump_prediction,
        "cp_pred",
        "Jump-aware regressor",
    ).set_index("surface")
    compared = response_jumps.join(
        jump_jumps,
        lsuffix="_response",
        rsuffix="_jump",
    )
    compared = compared[
        (compared["jump_size_response"] > JUMP_DETECTION_MIN_SIZE)
        | (compared["jump_size_jump"] > JUMP_DETECTION_MIN_SIZE)
    ]
    if compared.empty:
        return 0.0
    return float((compared["jump_x_jump"] - compared["jump_x_response"]).abs().max())


def model_jump_zone_rmse_disagreement(
    truth: pd.DataFrame,
    response_prediction: pd.DataFrame,
    jump_prediction: pd.DataFrame,
    width: float = 0.08,
) -> float:
    """Return model-vs-model RMSE near the supporting CFD jump zone."""

    jumps = pressure_jump_table(truth, "cp", "SU2")
    if jumps.empty:
        return float("nan")

    comparison = (
        truth[["PointID", "x", "surface"]]
        .merge(
            response_prediction[["PointID", "cp_pred"]].rename(
                columns={"cp_pred": "cp_response"}
            ),
            on="PointID",
            how="left",
        )
        .merge(
            jump_prediction[["PointID", "cp_pred"]].rename(
                columns={"cp_pred": "cp_jump"}
            ),
            on="PointID",
            how="left",
        )
    )
    mask = pd.Series(False, index=comparison.index)
    for row in jumps.itertuples(index=False):
        local_mask = (comparison["surface"] == row.surface) & (
            (comparison["x"] - float(row.jump_x)).abs() <= width
        )
        mask = mask | local_mask

    local = comparison[mask]
    if local.empty:
        return float("nan")
    residual = local["cp_response"].to_numpy(dtype=float) - local["cp_jump"].to_numpy(
        dtype=float
    )
    return float(np.sqrt(np.mean(residual**2)))


def full_curve_model_rmse_disagreement(
    response_prediction: pd.DataFrame,
    jump_prediction: pd.DataFrame,
) -> float:
    """Return whole-curve RMSE disagreement between the two surrogate models."""

    comparison = response_prediction[["PointID", "cp_pred"]].merge(
        jump_prediction[["PointID", "cp_pred"]],
        on="PointID",
        suffixes=("_response", "_jump"),
    )
    residual = comparison["cp_pred_response"].to_numpy(dtype=float) - comparison[
        "cp_pred_jump"
    ].to_numpy(dtype=float)
    return float(np.sqrt(np.mean(residual**2)))


def trust_risk_percent(
    value: float,
    scale: float,
) -> float:
    """Map a finite risk value onto a 0-100 indicator percentage."""

    if not np.isfinite(value):
        return 0.0
    return min(max(value / scale, 0.0), 1.0) * 100.0


def prediction_risk_summary(
    truth: pd.DataFrame,
    response_prediction: pd.DataFrame,
    jump_prediction: pd.DataFrame,
    normalized_distance: float,
    cfd_jump_span: float,
) -> dict[str, float | str]:
    """Compute comparable risk values for response-surface and jump-aware models.

    This is a presentation trust signal, not a calibrated uncertainty estimate.
    Local prediction errors should dominate the gauge. Neighborhood context
    such as distance to generated CFD cases and nearby jump movement is still
    shown, but it should not make a well-matched jump-aware prediction look
    risky by itself.
    """

    strong_jumps = pressure_jump_table(truth, "cp", "SU2")
    strong_jumps = strong_jumps[strong_jumps["jump_size"] > JUMP_DETECTION_MIN_SIZE]
    if strong_jumps.empty:
        response_rmse = full_curve_rmse_against_truth(truth, response_prediction)
        jump_rmse = full_curve_rmse_against_truth(truth, jump_prediction)
        model_gap = full_curve_model_rmse_disagreement(
            response_prediction,
            jump_prediction,
        )
        response_risk = max(
            trust_risk_percent(normalized_distance, SMOOTH_REGION_DISTANCE_RISK_SCALE),
            trust_risk_percent(response_rmse, SMOOTH_REGION_RMSE_RISK_SCALE),
        )
        jump_risk = max(
            trust_risk_percent(normalized_distance, SMOOTH_REGION_DISTANCE_RISK_SCALE),
            trust_risk_percent(jump_rmse, SMOOTH_REGION_RMSE_RISK_SCALE),
        )
        return {
            "mode": "Smooth region",
            "response_risk": response_risk,
            "jump_risk": jump_risk,
            "response_rmse": response_rmse,
            "jump_rmse": jump_rmse,
            "model_gap": model_gap,
            "jump_underperformance": max(0.0, jump_rmse - response_rmse),
            "cfd_jump_span": cfd_jump_span,
            "normalized_distance": normalized_distance,
            "response_jump_error": 0.0,
            "jump_error": 0.0,
        }

    response_error = max_jump_error_against_truth(truth, response_prediction)
    jump_error = max_jump_error_against_truth(truth, jump_prediction)
    response_size_error = max_jump_size_error_against_truth(truth, response_prediction)
    jump_size_error = max_jump_size_error_against_truth(truth, jump_prediction)
    response_rmse = jump_zone_rmse_against_truth(truth, response_prediction)
    jump_rmse = jump_zone_rmse_against_truth(truth, jump_prediction)
    jump_underperformance = max(0.0, jump_rmse - response_rmse)
    model_gap = model_jump_zone_rmse_disagreement(
        truth,
        response_prediction,
        jump_prediction,
    )
    shared_risk = max(
        trust_risk_percent(normalized_distance, RISK_DISTANCE_SCALE),
        trust_risk_percent(cfd_jump_span, RISK_CFD_JUMP_SPAN_SCALE),
    )
    response_risk = max(
        shared_risk,
        trust_risk_percent(response_error, RESPONSE_JUMP_ERROR_RISK_SCALE),
        trust_risk_percent(response_size_error, RESPONSE_JUMP_SIZE_RISK_SCALE),
        trust_risk_percent(response_rmse, RESPONSE_JUMP_RMSE_RISK_SCALE),
    )
    jump_risk = max(
        shared_risk,
        trust_risk_percent(jump_error, JUMP_AWARE_ERROR_RISK_SCALE),
        trust_risk_percent(jump_size_error, JUMP_AWARE_SIZE_RISK_SCALE),
        trust_risk_percent(jump_rmse, JUMP_AWARE_RMSE_RISK_SCALE),
        trust_risk_percent(
            jump_underperformance,
            JUMP_UNDERPERFORMANCE_RISK_SCALE,
        ),
    )

    return {
        "mode": "Jump region",
        "response_risk": response_risk,
        "jump_risk": jump_risk,
        "response_rmse": response_rmse,
        "jump_rmse": jump_rmse,
        "model_gap": model_gap,
        "jump_underperformance": jump_underperformance,
        "cfd_jump_span": cfd_jump_span,
        "normalized_distance": normalized_distance,
        "response_jump_error": response_error,
        "jump_error": jump_error,
    }


def render_dual_risk_viz(summary: dict[str, float | str]) -> None:
    """Render reusable dual-model risk bars and compact supporting metrics."""

    response_risk = float(summary["response_risk"])
    jump_risk = float(summary["jump_risk"])
    st.markdown(
        f"""
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;
    margin: 0.15rem 0 0.55rem 0;">
  <div>
    <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
      <strong>Response-surface risk</strong><span>{response_risk:.0f}%</span>
    </div>
    <div style="height: 12px; border-radius: 6px; margin-top: 0.35rem;
        background: linear-gradient(90deg, #2ecc71 0%, #f1c40f 55%, #e74c3c 100%);
        position: relative;">
      <div style="position: absolute; left: calc({response_risk:.1f}% - 2px);
          top: -4px; width: 4px; height: 20px; border-radius: 2px; background: white;
          box-shadow: 0 0 0 1px rgba(0,0,0,0.35);"></div>
    </div>
  </div>
  <div>
    <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
      <strong>Jump-aware risk</strong><span>{jump_risk:.0f}%</span>
    </div>
    <div style="height: 12px; border-radius: 6px; margin-top: 0.35rem;
        background: linear-gradient(90deg, #2ecc71 0%, #f1c40f 55%, #e74c3c 100%);
        position: relative;">
      <div style="position: absolute; left: calc({jump_risk:.1f}% - 2px);
          top: -4px; width: 4px; height: 20px; border-radius: 2px; background: white;
          box-shadow: 0 0 0 1px rgba(0,0,0,0.35);"></div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    render_compact_strip(
        [
            ("mode", str(summary["mode"])),
            ("dist", f"{float(summary['normalized_distance']):.3f}"),
            ("CFD span", f"{float(summary['cfd_jump_span']):.3f}"),
            ("response RMSE", f"{float(summary['response_rmse']):.3f}"),
            ("jump-aware RMSE", f"{float(summary['jump_rmse']):.3f}"),
            ("jump delta", f"{float(summary['jump_underperformance']):.3f}"),
            ("model gap", f"{float(summary['model_gap']):.3f}"),
        ]
    )


def render_curve_overlay(
    prediction: pd.DataFrame,
    truth: pd.DataFrame | None,
    truth_label: str,
    prediction_label: str,
    title: str,
    prediction_color: str = "#ff6b6b",
) -> None:
    """Render a pressure curve with consistent truth and surrogate styling."""

    plot_frames = []
    if truth is not None:
        truth_plot = truth.sort_values(["surface", "x"]).copy()
        truth_plot["source"] = truth_label
        truth_plot = truth_plot.rename(columns={"cp": "Cp"})
        plot_frames.append(truth_plot[["x", "surface", "Cp", "source"]])

    plotted = prediction.sort_values(["surface", "x"]).copy()
    plotted["source"] = prediction_label
    plotted = plotted.rename(columns={"cp_pred": "Cp"})
    plot_frames.append(plotted[["x", "surface", "Cp", "source"]])

    long = pd.concat(plot_frames, ignore_index=True)
    source_order = (
        [truth_label, prediction_label] if truth is not None else [prediction_label]
    )
    fig = px.line(
        long,
        x="x",
        y="Cp",
        color="source",
        line_dash="source",
        facet_row="surface",
        title=title,
        labels={"x": "x / chord", "source": ""},
        category_orders={"source": source_order},
        color_discrete_map={
            truth_label: "#74b9ff",
            prediction_label: prediction_color,
        },
        line_dash_map={
            truth_label: "solid",
            prediction_label: "dot",
        },
    )
    fig.update_yaxes(autorange="reversed", matches=None)
    fig.update_layout(height=500, legend_title_text="")
    st.plotly_chart(fig, width="stretch")


def render_multi_curve_overlay(
    truth: pd.DataFrame,
    truth_label: str,
    predictions: list[tuple[pd.DataFrame, str, str]],
    title: str,
) -> None:
    """Render SU2 truth with multiple surrogate curves on the same axes."""

    truth_plot = truth.sort_values(["surface", "x"]).copy()
    truth_plot["source"] = truth_label
    truth_plot = truth_plot.rename(columns={"cp": "Cp"})
    plot_frames = [truth_plot[["x", "surface", "Cp", "source"]]]

    color_map = {truth_label: "#74b9ff"}
    dash_map = {truth_label: "solid"}
    source_order = [truth_label]

    for prediction, label, color in predictions:
        plotted = prediction.sort_values(["surface", "x"]).copy()
        plotted["source"] = label
        plotted = plotted.rename(columns={"cp_pred": "Cp"})
        plot_frames.append(plotted[["x", "surface", "Cp", "source"]])
        color_map[label] = color
        dash_map[label] = "dot"
        source_order.append(label)

    long = pd.concat(plot_frames, ignore_index=True)
    fig = px.line(
        long,
        x="x",
        y="Cp",
        color="source",
        line_dash="source",
        facet_row="surface",
        title=title,
        labels={"x": "x / chord", "source": ""},
        category_orders={"source": source_order},
        color_discrete_map=color_map,
        line_dash_map=dash_map,
    )
    fig.update_yaxes(autorange="reversed", matches=None)
    fig.update_layout(height=500, legend_title_text="")
    st.plotly_chart(fig, width="stretch")


def render_surface_comparison(
    predictions: pd.DataFrame,
    prediction_label: str,
    prediction_color: str,
) -> None:
    """Render SU2-vs-surrogate pressure coefficient curves."""

    render_curve_overlay(
        prediction=predictions,
        truth=predictions,
        truth_label="SU2 holdout",
        prediction_label=prediction_label,
        prediction_color=prediction_color,
        title="Surface pressure coefficient: SU2 holdout vs surrogate",
    )


def render_residuals(predictions: pd.DataFrame) -> None:
    """Render prediction error along the airfoil surface."""

    fig = px.line(
        predictions,
        x="x",
        y="error",
        color="surface",
        title="Residual error along the airfoil surface",
        labels={
            "x": "x / chord",
            "error": "Surrogate Cp - SU2 Cp",
        },
    )
    fig.add_hline(y=0.0, line_width=1, line_dash="dash", line_color="gray")
    fig.update_layout(height=360, legend_title_text="")
    st.plotly_chart(fig, width="stretch")


def model_metrics_table(
    response: SurrogateResult,
    jump_aware: JumpAwareResult,
) -> pd.DataFrame:
    """Build a side-by-side holdout metric table for both surrogate families."""

    response_jump_zone = compute_jump_zone_metrics(response.predictions)
    rows = [
        {
            "Model": "Response surface",
            "R2": response.metrics["R2"],
            "MAE Cp": response.metrics["MAE"],
            "RMSE Cp": response.metrics["RMSE"],
            "Jump-zone MAE": response_jump_zone["MAE"],
            "Jump-zone RMSE": response_jump_zone["RMSE"],
            "Curve time ms": response.inference_seconds * 1000.0,
            "Fit/build time": response.fit_seconds,
        },
        {
            "Model": "Jump-aware regressor",
            "R2": jump_aware.metrics["R2"],
            "MAE Cp": jump_aware.metrics["MAE"],
            "RMSE Cp": jump_aware.metrics["RMSE"],
            "Jump-zone MAE": jump_aware.jump_zone_metrics["MAE"],
            "Jump-zone RMSE": jump_aware.jump_zone_metrics["RMSE"],
            "Curve time ms": jump_aware.inference_seconds * 1000.0,
            "Fit/build time": jump_aware.fit_seconds,
        },
    ]
    return pd.DataFrame(rows)


def model_jump_table(
    response_predictions: pd.DataFrame,
    jump_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Build jump-location comparison rows for both surrogate families."""

    response = jump_comparison(response_predictions)
    response["Model"] = "Response surface"
    jump_aware = jump_comparison(jump_predictions)
    jump_aware["Model"] = "Jump-aware regressor"
    combined = pd.concat([response, jump_aware], ignore_index=True)
    return combined[
        [
            "Model",
            "surface",
            "jump_x_su2",
            "jump_x_surrogate",
            "jump_x_error",
            "jump_size_su2",
            "jump_size_surrogate",
        ]
    ].rename(
        columns={
            "surface": "Surface",
            "jump_x_su2": "SU2 jump x",
            "jump_x_surrogate": "Model jump x",
            "jump_x_error": "x error",
            "jump_size_su2": "SU2 jump",
            "jump_size_surrogate": "Model jump",
        }
    )


def render_model_jump_alignment(
    response_predictions: pd.DataFrame,
    jump_predictions: pd.DataFrame,
) -> None:
    """Render the shared dual-model risk visualization for holdout validation."""

    summary = prediction_risk_summary(
        truth=response_predictions,
        response_prediction=response_predictions,
        jump_prediction=jump_predictions,
        normalized_distance=0.0,
        cfd_jump_span=0.0,
    )
    render_dual_risk_viz(summary)


def render_custom_prediction(
    prediction: pd.DataFrame,
    truth: pd.DataFrame | None = None,
    truth_label: str = "Nearest SU2 case",
    prediction_label: str = "Surrogate",
    prediction_color: str = "#ff6b6b",
) -> None:
    """Render an interactive surrogate prediction, with SU2 truth when available."""

    render_curve_overlay(
        prediction=prediction,
        truth=truth,
        truth_label=truth_label,
        prediction_label=prediction_label,
        prediction_color=prediction_color,
        title="Predicted surface-pressure curve for selected condition",
    )


def render_history(history: pd.DataFrame, columns: list[str], title: str) -> None:
    """Render solver history columns with Plotly."""

    available = [column for column in columns if column in history]
    if not available:
        st.info("No matching history columns found.")
        return

    long = history[["Inner_Iter", *available]].melt(
        id_vars="Inner_Iter",
        var_name="quantity",
        value_name="value",
    )
    fig = px.line(
        long,
        x="Inner_Iter",
        y="value",
        color="quantity",
        title=title,
        markers=False,
    )
    fig.update_layout(height=390, legend_title_text="")
    st.plotly_chart(fig, width="stretch")


def render_mesh(mesh: MeshData) -> None:
    """Render mesh marker geometry."""

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(
        mesh.farfield["x"],
        mesh.farfield["y"],
        s=2,
        alpha=0.45,
        label="farfield marker",
    )
    ax.plot(
        mesh.airfoil["x"],
        mesh.airfoil["y"],
        linewidth=2,
        label="airfoil marker",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Boundary markers from the SU2 mesh")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def render_field_contour(
    flow_field: FlowField,
    mesh: MeshData,
    field_name: str,
    title: str,
    cmap: str,
    limits: tuple[float, float, float, float],
) -> None:
    """Render a ParaView-like filled field view from the VTU output."""

    points = flow_field.points
    scalar = flow_field.fields[field_name]
    if scalar.ndim > 1:
        scalar = np.linalg.norm(scalar, axis=1)

    triangulation = mtri.Triangulation(
        points["x"].to_numpy(),
        points["y"].to_numpy(),
        flow_field.triangles,
    )

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    colored = ax.tripcolor(
        triangulation,
        scalar,
        shading="gouraud",
        cmap=cmap,
    )
    ax.tricontour(
        triangulation,
        scalar,
        colors="black",
        linewidths=0.25,
        alpha=0.25,
        levels=18,
    )
    ax.plot(mesh.airfoil["x"], mesh.airfoil["y"], color="black", linewidth=1.3)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(colored, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def render_adjoint_surface(surface_adjoint: pd.DataFrame) -> None:
    """Render adjoint surface sensitivity."""

    plotted = surface_adjoint.copy()
    plotted["surface"] = np.where(plotted["y"] >= 0.0, "upper", "lower")
    plotted = plotted.sort_values(["surface", "x"])

    sensitivity_fig = px.line(
        plotted,
        x="x",
        y="Surface_Sensitivity",
        color="surface",
        title="Surface sensitivity of drag objective",
        labels={"Surface_Sensitivity": "Surface sensitivity", "x": "x / chord"},
    )
    sensitivity_fig.update_layout(height=380, legend_title_text="")
    st.plotly_chart(sensitivity_fig, width="stretch")


def render_log_excerpt(path: Path, title: str) -> None:
    """Show the end of an SU2 log file."""

    lines = path.read_text(errors="replace").splitlines()
    st.markdown(f"**{title}**")
    st.code("\n".join(lines[-24:]), language="text")


def main() -> None:
    """Render the Streamlit app."""

    st.set_page_config(
        page_title="Airfoil CFD-to-Surrogate Study",
        page_icon=None,
        layout="wide",
    )
    apply_layout_styles()

    artifacts = load_case_artifacts()
    version_token = data_version_token()
    surface, summary = load_sweep_data(version_token)
    jump_catalog = load_jump_catalog(version_token)
    summary_by_case = summary.set_index("case_id")

    st.title("Airfoil CFD-to-Surrogate Study")
    st.caption(
        "A progression from the SU2 NACA 0012 quick-start case to a dense "
        "Mach/AoA sweep and a learned surface-pressure surrogate."
    )

    case_ids = summary.sort_values(["mach", "aoa_deg"])["case_id"].to_list()
    default_holdout_row = summary_by_case.loc[
        (
            DEFAULT_HOLDOUT_CASE
            if DEFAULT_HOLDOUT_CASE in case_ids
            else case_ids[len(case_ids) // 2]
        )
    ]

    st.sidebar.markdown("**Holdout case**")
    holdout_mach_values, _holdout_aoa_values = generated_value_options(summary)
    selected_holdout_mach = st.sidebar.select_slider(
        "Mach",
        options=holdout_mach_values,
        value=float(default_holdout_row["mach"]),
        format_func=lambda value: f"{value:.4f}",
        key="holdout_mach_v3",
    )
    holdout_aoa_values = available_aoa_options(summary, selected_holdout_mach)
    default_holdout_aoa = float(default_holdout_row["aoa_deg"])
    if default_holdout_aoa not in holdout_aoa_values:
        default_holdout_aoa = holdout_aoa_values[len(holdout_aoa_values) // 2]
    selected_holdout_aoa = st.sidebar.select_slider(
        "AoA (deg)",
        options=holdout_aoa_values,
        value=default_holdout_aoa,
        format_func=lambda value: f"{value:+.3f}",
        key="holdout_aoa_v3",
    )
    selected_holdout = exact_case_id(
        summary,
        selected_holdout_mach,
        selected_holdout_aoa,
    )
    if selected_holdout is None:
        selected_holdout = case_ids[len(case_ids) // 2]
    selected_case_row = summary_by_case.loc[selected_holdout]
    st.sidebar.caption(format_case(summary_by_case, selected_holdout))

    st.sidebar.markdown("**Sweep envelope**")
    st.sidebar.caption(
        f"Mach {summary['mach'].min():.3f}-{summary['mach'].max():.3f}; "
        f"AoA {summary['aoa_deg'].min():+.2f}-{summary['aoa_deg'].max():+.2f} deg"
    )
    st.sidebar.caption(
        "SU2 timings were measured on the local machine that generated the "
        "sweep. Surrogate timings are measured in the active Streamlit runtime."
    )
    st.sidebar.caption(
        "Changing the holdout case invalidates that leave-one-case-out run and "
        "rebuilds the validation output for the newly selected CFD case."
    )

    view_labels = [
        "1. CFD sweep",
        "2. Holdout comparison",
        "3. Design explorer",
        "4. Field visuals",
        "5. Solver artifacts",
        "6. Notes",
    ]
    selected_view = st.segmented_control(
        "View",
        view_labels,
        default=view_labels[0],
        label_visibility="collapsed",
        width="stretch",
    )
    if selected_view is None:
        selected_view = view_labels[0]

    if selected_view == view_labels[0]:
        st.subheader("From one solver case to a dense operating-condition sweep")
        st.caption(
            "SU2-generated pressure curves over a dense Mach/AoA grid. The map "
            "shows where the dominant pressure jump moves across the design space."
        )

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("CFD cases", f"{summary['case_id'].nunique():,}")
        col_b.metric("Surface samples", f"{len(surface):,}")
        col_c.metric("Mach values", f"{summary['mach'].nunique():,}")
        col_d.metric("AoA values", f"{summary['aoa_deg'].nunique():,}")

        st.markdown("**Pressure-jump location across the generated sweep**")
        render_jump_map(jump_catalog)
        with st.expander("How to read the pressure-jump map"):
            st.write(
                "`jump x / chord` is the detected chordwise position of the "
                "steepest interior pressure-coefficient change on each airfoil "
                "surface. Smooth regions are easier to interpolate. Sharp color "
                "transitions mark operating regions where small Mach/AoA changes "
                "move the jump, which is where a surrogate can smear or shift the "
                "drop."
            )

    elif selected_view == view_labels[1]:
        st.subheader("Held-out CFD case: response surface vs jump-aware regressor")
        st.caption(
            "Leave-one-case-out validation. Blue is the withheld SU2 curve; red "
            "is the response surface; orange is the jump-aware regressor."
        )

        response_surrogate = train_holdout_surrogate(
            selected_holdout,
            version_token,
            MODEL_VERSION,
        )
        jump_surrogate = train_jump_aware_holdout_surrogate(
            selected_holdout,
            version_token,
            MODEL_VERSION,
        )
        solver_seconds = solver_time_for_case(summary, selected_holdout)
        holdout_row = summary_by_case.loc[selected_holdout]

        metrics_view = model_metrics_table(response_surrogate, jump_surrogate)
        metrics_display = metrics_view[
            [
                "Model",
                "R2",
                "RMSE Cp",
                "Jump-zone RMSE",
                "Curve time ms",
                "Fit/build time",
            ]
        ]
        render_compact_strip(
            [
                (
                    "holdout",
                    (
                        f"Mach {holdout_row['mach']:.4f}, "
                        f"AoA {holdout_row['aoa_deg']:+.3f} deg"
                    ),
                ),
                ("SU2 case", f"{solver_seconds:.3f}s"),
                ("response RMSE", f"{response_surrogate.metrics['RMSE']:.3f}"),
                ("jump-aware RMSE", f"{jump_surrogate.metrics['RMSE']:.3f}"),
                (
                    "jump-aware curve",
                    f"{jump_surrogate.inference_seconds * 1000:.1f}ms",
                ),
            ]
        )

        st.markdown("**Validation risk**")
        render_model_jump_alignment(
            response_surrogate.predictions,
            jump_surrogate.predictions,
        )
        if is_boundary_case(summary, selected_holdout):
            st.warning(
                "This holdout is on the edge of the generated Mach/AoA envelope. "
                "Once the selected CFD case is withheld, validation becomes "
                "one-sided near that boundary and errors can be much larger.",
                icon=None,
            )

        render_multi_curve_overlay(
            truth=response_surrogate.predictions,
            truth_label="SU2 holdout",
            predictions=[
                (
                    response_surrogate.predictions,
                    "Response-surface surrogate",
                    "#ff6b6b",
                ),
                (
                    jump_surrogate.predictions,
                    "Jump-aware regressor",
                    "#f4a340",
                ),
            ],
            title="Surface pressure coefficient: SU2 holdout vs both surrogates",
        )

        with st.expander("Accuracy and timing details"):
            st.dataframe(
                metrics_display.style.format(
                    {
                        "R2": "{:.4f}",
                        "RMSE Cp": "{:.4f}",
                        "Jump-zone RMSE": "{:.4f}",
                        "Curve time ms": "{:.2f}",
                        "Fit/build time": "{:.4f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Timing comparison is order-of-magnitude: SU2 was measured during "
                "local data generation; surrogate timing is measured in the active "
                "Streamlit runtime."
            )

        col_left, col_right = st.columns([1.25, 1])
        with col_left:
            st.markdown("**Pressure-jump check**")
            st.dataframe(
                model_jump_table(
                    response_surrogate.predictions,
                    jump_surrogate.predictions,
                ),
                width="stretch",
                hide_index=True,
            )
        with col_right:
            st.markdown("**Jump-aware training signal**")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Weighted rows", f"{jump_surrogate.model.weighted_rows:,}")
            col_b.metric(
                "Mean weight",
                f"{jump_surrogate.model.mean_sample_weight:.2f}x",
            )
            col_c.metric("Max weight", f"{jump_surrogate.model.max_sample_weight:.1f}x")
            st.write(
                "The jump-aware regressor uses the same CFD cases, but gives "
                "higher training weight to the pressure-jump center and to the "
                "upstream and downstream shoulder points where the curve changes "
                "regime."
            )

        with st.expander("How the jump-aware regressor is trained"):
            st.write(
                "The regressor uses engineered numeric features: Mach, angle of "
                "attack, x/y surface location, upper/lower surface, and simple "
                "interaction terms such as Mach x x and AoA x x. It is trained "
                "with an Extra Trees ensemble because tree splits can preserve "
                "localized sharp changes better than a globally smooth fit."
            )
            st.write(
                "Sample weights are assigned per surface point. For each CFD "
                "case and airfoil side, the app detects the strongest interior "
                "pressure jump. Only jumps above the minimum-strength threshold "
                "are used as jump-location features, which avoids treating tiny "
                "smooth-region gradients as real shock anchors. It then estimates "
                "jump-relative features and adds a stronger, narrow Gaussian "
                "weight boost around the jump center plus secondary boosts around "
                "the upstream and downstream shoulders. Smooth regions still "
                "count, but errors around the whole break region count more "
                "heavily."
            )
            st.write(
                f"Current weighting: base weight `1.0`, center width "
                f"`{JUMP_WEIGHT_WIDTH:.3f}` chord, shoulder offset "
                f"`{JUMP_SHOULDER_OFFSET:.3f}` chord, shoulder width "
                f"`{JUMP_SHOULDER_WIDTH:.3f}` chord, all scaled by detected "
                "jump strength."
            )

    elif selected_view == view_labels[2]:
        st.subheader("Interactive design-space exploration")
        st.caption(
            "This view trains both models on the full sweep and predicts the "
            "pressure curve for a selected condition. The nearest generated SU2 "
            "case is overlaid so continuity around known CFD samples is visible."
        )

        full_response_surrogate = train_full_surrogate(version_token, MODEL_VERSION)
        full_jump_surrogate = train_full_jump_aware_surrogate(
            version_token,
            MODEL_VERSION,
        )
        surface_template = surface[surface["case_id"] == selected_holdout].copy()

        col_a, col_b = st.columns(2)
        with col_a:
            selected_mach = st.slider(
                "Mach number",
                min_value=0.6500,
                max_value=0.9250,
                value=float(selected_case_row["mach"]),
                step=0.0025,
                format="%.4f",
                key="combined_explorer_mach",
            )
        with col_b:
            selected_aoa = st.slider(
                "Angle of attack (deg)",
                min_value=-2.000,
                max_value=3.500,
                value=float(selected_case_row["aoa_deg"]),
                step=0.025,
                format="%.3f",
                key="combined_explorer_aoa",
            )

        status_label, status_text = interpolation_status(
            summary,
            selected_mach,
            selected_aoa,
        )
        render_status_pill(
            status_label,
            status_text,
            is_ok=status_label == "Interpolation",
        )

        response_prediction, response_seconds = predict_surface_curve(
            full_response_surrogate.model,
            surface_template,
            selected_mach,
            selected_aoa,
        )
        jump_prediction, jump_seconds = predict_jump_aware_curve(
            full_jump_surrogate.model,
            surface_template,
            selected_mach,
            selected_aoa,
        )

        nearest = nearest_design_cases(summary, selected_mach, selected_aoa)
        nearest_case = nearest.iloc[0]
        nearest_case_id = str(nearest_case["case_id"])
        truth = surface[surface["case_id"] == nearest_case_id]
        nearest_seconds = float(nearest.iloc[0]["runtime_seconds"])
        nearest_label = (
            f"Nearest SU2 case: Mach {nearest_case['mach']:.4f}, "
            f"AoA {nearest_case['aoa_deg']:+.3f} deg"
        )

        explorer_metrics = pd.DataFrame(
            [
                {
                    "Model": "Response surface",
                    "Prediction ms": response_seconds * 1000.0,
                    "Nearest-case speedup": safe_speedup(
                        nearest_seconds,
                        response_seconds,
                    ),
                    "Fit/build time": full_response_surrogate.fit_seconds,
                },
                {
                    "Model": "Jump-aware regressor",
                    "Prediction ms": jump_seconds * 1000.0,
                    "Nearest-case speedup": safe_speedup(
                        nearest_seconds,
                        jump_seconds,
                    ),
                    "Fit/build time": full_jump_surrogate.fit_seconds,
                },
            ]
        )
        render_compact_strip(
            [
                ("nearest", nearest_label.replace("Nearest SU2 case: ", "")),
                ("distance", f"{nearest_case['normalized_distance']:.4f}"),
                ("response", f"{response_seconds * 1000:.1f}ms"),
                ("jump-aware", f"{jump_seconds * 1000:.1f}ms"),
                ("SU2 case", f"{nearest_seconds:.2f}s"),
            ]
        )

        jump_context = nearest_jump_context(jump_catalog, nearest)
        cfd_jump_span = (
            float(jump_context["jump_x_span"].max()) if not jump_context.empty else 0.0
        )
        st.markdown("**Prediction trust signal**")
        render_dual_risk_viz(
            prediction_risk_summary(
                truth=truth,
                response_prediction=response_prediction,
                jump_prediction=jump_prediction,
                normalized_distance=float(nearest_case["normalized_distance"]),
                cfd_jump_span=cfd_jump_span,
            )
        )

        render_multi_curve_overlay(
            truth=truth,
            truth_label=nearest_label,
            predictions=[
                (
                    response_prediction,
                    "Response-surface surrogate",
                    "#ff6b6b",
                ),
                (
                    jump_prediction,
                    "Jump-aware regressor",
                    "#f4a340",
                ),
            ],
            title="Predicted surface-pressure curve for selected condition",
        )

        with st.expander("Prediction timing and speedup details"):
            st.dataframe(
                explorer_metrics.style.format(
                    {
                        "Prediction ms": "{:.2f}",
                        "Nearest-case speedup": "{:.0f}x",
                        "Fit/build time": "{:.4f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        st.markdown("**Nearest supporting CFD cases**")
        nearest_view = nearest[
            [
                "case_label",
                "runtime_seconds",
                "iterations",
                "cp_range",
                "normalized_distance",
            ]
        ].rename(
            columns={
                "case_label": "Case",
                "runtime_seconds": "SU2 seconds",
                "iterations": "Iterations",
                "cp_range": "Cp range",
                "normalized_distance": "Normalized distance",
            }
        )
        st.dataframe(nearest_view, width="stretch", hide_index=True)

        with st.expander("How to read this view"):
            st.write(
                "The response surface can reproduce an exact generated CFD case "
                "because the full-sweep explorer includes it in the interpolation "
                "grid. The jump-aware regressor is always a learned prediction. "
                "Inside the sweep envelope this is interpolation; outside it is "
                "extrapolation and should trigger more CFD validation."
            )

    elif selected_view == view_labels[3]:
        st.subheader("ParaView-like field visuals from the reference SU2 run")
        st.caption(
            "The surrogate tabs focus on surface pressure because that is compact "
            "enough for a tractable CFD sweep. This tab keeps one full-field VTU "
            "result visible, showing the kind of spatial field a richer surrogate "
            "would eventually need to approximate."
        )

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("VTU points", f"{artifacts.flow_field.point_count:,}")
        col_b.metric("VTU cells", f"{artifacts.flow_field.cell_count:,}")
        col_c.metric("Triangles used", f"{len(artifacts.flow_field.triangles):,}")
        col_d.metric("Point fields", f"{len(artifacts.flow_field.fields):,}")

        field_names = sorted(artifacts.flow_field.fields.keys())
        selected_field = st.selectbox(
            "Field",
            field_names,
            index=field_names.index("Mach") if "Mach" in field_names else 0,
        )
        view_mode = st.radio(
            "View",
            ["Airfoil zoom", "Full farfield"],
            horizontal=True,
        )
        limits = (
            (-0.20, 1.25, -0.45, 0.45)
            if view_mode == "Airfoil zoom"
            else (-4.5, 5.5, -5.0, 5.0)
        )
        cmap = "viridis" if selected_field != "Pressure_Coefficient" else "coolwarm"

        render_field_contour(
            flow_field=artifacts.flow_field,
            mesh=artifacts.mesh,
            field_name=selected_field,
            title=f"{selected_field} from flow.vtu",
            cmap=cmap,
            limits=limits,
        )

        col_left, col_right = st.columns(2)
        with col_left:
            render_field_contour(
                flow_field=artifacts.flow_field,
                mesh=artifacts.mesh,
                field_name="Mach",
                title="Mach contours around the airfoil",
                cmap="viridis",
                limits=(-0.20, 1.25, -0.45, 0.45),
            )
        with col_right:
            render_field_contour(
                flow_field=artifacts.flow_field,
                mesh=artifacts.mesh,
                field_name="Pressure_Coefficient",
                title="Pressure coefficient contours around the airfoil",
                cmap="coolwarm",
                limits=(-0.20, 1.25, -0.45, 0.45),
            )

    elif selected_view == view_labels[4]:
        st.subheader("Solver artifacts behind the dataset")
        st.write(
            "The app reads committed SU2 outputs, so the viewer does not need "
            "SU2 installed. The artifacts remain inspectable: config, mesh, "
            "convergence history, direct-flow fields, and adjoint sensitivity."
        )

        direct_config = artifacts.direct_config
        adjoint_config = artifacts.adjoint_config
        mesh = artifacts.mesh

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Reference Mach", direct_config["MACH_NUMBER"])
        col_b.metric("Reference AoA", f"{direct_config['AOA']} deg")
        col_c.metric("Mesh points", f"{mesh.point_count:,}")
        col_d.metric("Mesh cells", f"{mesh.element_count:,}")

        with st.expander("SU2 quick-start configuration", expanded=True):
            setup_rows = [
                {"Setting": "Direct problem", "Value": direct_config["MATH_PROBLEM"]},
                {"Setting": "Adjoint problem", "Value": adjoint_config["MATH_PROBLEM"]},
                {"Setting": "Objective", "Value": direct_config["OBJECTIVE_FUNCTION"]},
                {"Setting": "Mesh", "Value": direct_config["MESH_FILENAME"]},
                {"Setting": "Wall marker", "Value": direct_config["MARKER_EULER"]},
                {"Setting": "Farfield marker", "Value": direct_config["MARKER_FAR"]},
                {"Setting": "Iterations", "Value": direct_config["ITER"]},
                {"Setting": "Output files", "Value": direct_config["OUTPUT_FILES"]},
            ]
            st.dataframe(pd.DataFrame(setup_rows), width="stretch", hide_index=True)
            st.markdown(f"[SU2 Quick Start]({SU2_QUICK_START_URL})")
            st.markdown(f"[SU2 v8.5.0 release]({SU2_RELEASE_URL})")

        with st.expander("Mesh markers"):
            col_left, col_right = st.columns([1.2, 1])
            with col_left:
                render_mesh(mesh)
            with col_right:
                st.markdown("**Marker counts**")
                marker_rows = [
                    {"Marker": marker, "Edges": count}
                    for marker, count in mesh.marker_counts.items()
                ]
                st.dataframe(
                    pd.DataFrame(marker_rows),
                    width="stretch",
                    hide_index=True,
                )

        with st.expander("Direct solver convergence"):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Iterations", f"{len(artifacts.direct_history):,}")
            col_b.metric(
                "Final rho residual",
                value_or_dash(latest_value(artifacts.direct_history, "rms[Rho]")),
            )
            col_c.metric(
                "Final rhoE residual",
                value_or_dash(latest_value(artifacts.direct_history, "rms[RhoE]")),
            )
            render_history(
                artifacts.direct_history,
                ["rms[Rho]", "rms[RhoU]", "rms[RhoV]", "rms[RhoE]"],
                "Direct solver residual history",
            )
            render_log_excerpt(DIRECT_LOG, "Direct solve")

        with st.expander("Adjoint sensitivity"):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Iterations", f"{len(artifacts.adjoint_history):,}")
            col_b.metric(
                "Final adjoint rho residual",
                value_or_dash(latest_value(artifacts.adjoint_history, "rms[A_Rho]")),
            )
            col_c.metric(
                "Final Sens_AoA",
                value_or_dash(
                    latest_value(artifacts.adjoint_history, "Sens_AoA"),
                    6,
                ),
            )
            render_history(
                artifacts.adjoint_history,
                ["rms[A_Rho]", "rms[A_RhoU]", "rms[A_RhoV]", "rms[A_E]"],
                "Adjoint solver residual history",
            )
            render_adjoint_surface(artifacts.surface_adjoint)
            render_log_excerpt(ADJOINT_LOG, "Adjoint solve")

    elif selected_view == view_labels[5]:
        st.subheader("Engineering notes")
        st.markdown(
            """
**What this study adds over a tabular surrogate demo**

- The training data is generated by a physics solver instead of coming from a
  static public table.
- The target is a spatial distribution over an airfoil surface, not one scalar.
- The scope is deliberately bounded: fixed NACA 0012 geometry, Mach/AoA
  variation, and pressure coefficient on a 2D airfoil surface.
- Evaluation uses a withheld Mach/AoA case, which is closer to the question:
  can the model interpolate to a new operating condition?
- The interface shows timing, accuracy, residuals, and distance to training data
  together because trust is a product concern, not just an ML metric.
- A second jump-aware regressor uses estimated jump-location features and
  sample weights near the jump center and shoulders, making the training
  objective more aligned with a physically important local feature than plain
  global error alone.

**How this relates to aerodynamic surrogate products**

Industrial aerodynamic surrogate workflows often target pressure-field
prediction moving from solver-scale timing to millisecond-scale model inference,
with accuracy reporting that goes beyond a single global score. This study is a
bounded 2D analogue: it does not reproduce a production system, but it exercises
the same shape of problem at a tractable scale.

The next level of realism would be geometry variation, richer representations
such as meshes or CAD boundary representations, full-field targets, uncertainty
estimation, and a validation workflow that clearly separates interpolation from
extrapolation.
            """
        )
        st.markdown("**Local artifacts**")
        artifact_rows = [
            {"Artifact": "Direct config", "Path": str(DIRECT_CONFIG.relative_to(ROOT))},
            {
                "Artifact": "Adjoint config",
                "Path": str(ADJOINT_CONFIG.relative_to(ROOT)),
            },
            {"Artifact": "Mesh", "Path": str(MESH_PATH.relative_to(ROOT))},
            {
                "Artifact": "Reference flow field",
                "Path": str(FLOW_VTU.relative_to(ROOT)),
            },
            {"Artifact": "Sweep summary", "Path": str(SWEEP_SUMMARY.relative_to(ROOT))},
            {
                "Artifact": "Sweep surface data",
                "Path": str(SWEEP_SURFACE.relative_to(ROOT)),
            },
            {"Artifact": "Sweep generator", "Path": "scripts/generate_sweep.py"},
            {"Artifact": "Single-case rerunner", "Path": "scripts/run_su2.sh"},
        ]
        st.dataframe(pd.DataFrame(artifact_rows), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
