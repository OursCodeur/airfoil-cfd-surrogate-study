"""Run a Mach/AoA SU2 sweep and build app-ready surface-pressure data."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input"
SWEEP_RESULTS_DIR = ROOT / "data" / "results" / "sweep"
CONFIG_PATH = INPUT_DIR / "inv_NACA0012.cfg"
MESH_PATH = INPUT_DIR / "mesh_NACA0012_inv.su2"

MACH_VALUES = np.linspace(0.7000, 0.8875, 25)
AOA_VALUES = np.linspace(-1.250, 3.125, 20)


def case_id(mach: float, aoa: float) -> str:
    """Return a stable case identifier for filenames and joins."""

    aoa_label = f"{aoa:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    mach_label = f"{mach:.4f}".replace(".", "p")
    return f"mach_{mach_label}_aoa_{aoa_label}"


def design_cases() -> list[tuple[float, float]]:
    """Return the uniform dense Mach/AoA sweep."""

    return [(float(mach), float(aoa)) for mach in MACH_VALUES for aoa in AOA_VALUES]


def update_config(template: str, mach: float, aoa: float) -> str:
    """Patch the official quick-start config for one sweep condition."""

    replacements = {
        "MACH_NUMBER": f"{mach:.6f}",
        "AOA": f"{aoa:.6f}",
        "OUTPUT_FILES": "(SURFACE_CSV)",
    }

    lines = []
    for line in template.splitlines():
        stripped = line.strip()
        key = stripped.split("=", maxsplit=1)[0].strip() if "=" in stripped else ""
        if key in replacements:
            lines.append(f"{key}= {replacements[key]}")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def read_su2_csv(path: Path) -> pd.DataFrame:
    """Read a SU2 CSV with normalized column names."""

    df = pd.read_csv(path)
    df.columns = [column.strip().strip('"') for column in df.columns]
    return df


def compute_pressure_coefficient(surface: pd.DataFrame, mach: float) -> pd.Series:
    """Compute Cp from SU2 conservative variables."""

    gamma = 1.4
    density = surface["Density"]
    momentum_x = surface["Momentum_x"]
    momentum_y = surface["Momentum_y"]
    energy = surface["Energy"]

    velocity_squared = (momentum_x**2 + momentum_y**2) / density**2
    pressure = (gamma - 1.0) * (energy - 0.5 * density * velocity_squared)

    freestream_pressure = 101_325.0
    freestream_temperature = 273.15
    gas_constant = 287.058
    speed_of_sound = np.sqrt(gamma * gas_constant * freestream_temperature)
    freestream_velocity = mach * speed_of_sound
    freestream_density = freestream_pressure / (gas_constant * freestream_temperature)
    dynamic_pressure = 0.5 * freestream_density * freestream_velocity**2

    return (pressure - freestream_pressure) / dynamic_pressure


def final_value(df: pd.DataFrame, column: str) -> float:
    """Return the final finite value from a history column."""

    if column not in df:
        return float("nan")
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[-1])


def run_case(
    su2_cfd: str,
    template: str,
    run_root: Path,
    mach: float,
    aoa: float,
    threads: int,
    reuse_existing: bool,
    previous_runtime_seconds: float | None,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Run one SU2 case and return surface data plus summary metrics."""

    cid = case_id(mach, aoa)
    case_dir = run_root / cid
    case_dir.mkdir(parents=True, exist_ok=True)

    history_path = case_dir / "history.csv"
    surface_path = case_dir / "surface_flow.csv"
    if reuse_existing and history_path.exists() and surface_path.exists():
        runtime_seconds = (
            previous_runtime_seconds
            if previous_runtime_seconds is not None
            else float("nan")
        )
    else:
        shutil.copy2(MESH_PATH, case_dir / MESH_PATH.name)
        (case_dir / CONFIG_PATH.name).write_text(update_config(template, mach, aoa))

        command = [su2_cfd, "-t", str(threads), CONFIG_PATH.name]
        started_at = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=case_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        runtime_seconds = time.perf_counter() - started_at
        (case_dir / "direct.log").write_text(completed.stdout)
        if completed.returncode != 0:
            raise RuntimeError(f"SU2 failed for {cid}. See {case_dir / 'direct.log'}")

    history = read_su2_csv(history_path)
    surface = read_su2_csv(surface_path)
    surface["case_id"] = cid
    surface["mach"] = mach
    surface["aoa_deg"] = aoa
    surface["surface"] = np.where(surface["y"] >= 0.0, "upper", "lower")
    surface["cp"] = compute_pressure_coefficient(surface, mach)
    surface["velocity_magnitude"] = np.sqrt(
        (surface["Momentum_x"] / surface["Density"]) ** 2
        + (surface["Momentum_y"] / surface["Density"]) ** 2
    )

    summary = {
        "case_id": cid,
        "mach": mach,
        "aoa_deg": aoa,
        "iterations": len(history),
        "surface_rows": len(surface),
        "runtime_seconds": runtime_seconds,
        "final_rms_rho": final_value(history, "rms[Rho]"),
        "final_rms_rhoe": final_value(history, "rms[RhoE]"),
        "cp_min": float(surface["cp"].min()),
        "cp_max": float(surface["cp"].max()),
        "cp_range": float(surface["cp"].max() - surface["cp"].min()),
    }
    return surface, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--su2-cfd", default="SU2_CFD")
    parser.add_argument("--run-root", type=Path, default=ROOT / "run" / "sweep")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing per-case run outputs and run only missing cases.",
    )
    args = parser.parse_args()

    template = CONFIG_PATH.read_text()
    args.run_root.mkdir(parents=True, exist_ok=True)
    SWEEP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    previous_runtime_by_case: dict[str, float] = {}
    if SWEEP_RESULTS_DIR.joinpath("sweep_summary.csv").exists():
        previous_summary = read_su2_csv(SWEEP_RESULTS_DIR / "sweep_summary.csv")
        previous_runtime_by_case = dict(
            zip(
                previous_summary["case_id"],
                previous_summary["runtime_seconds"],
                strict=True,
            )
        )

    surfaces = []
    summaries = []
    cases = design_cases()
    total = len(cases)
    for case_number, (mach, aoa) in enumerate(cases, start=1):
        cid = case_id(mach, aoa)
        print(f"[{case_number:03d}/{total}] preparing {cid}")
        surface, summary = run_case(
            su2_cfd=args.su2_cfd,
            template=template,
            run_root=args.run_root,
            mach=mach,
            aoa=aoa,
            threads=args.threads,
            reuse_existing=args.reuse_existing,
            previous_runtime_seconds=previous_runtime_by_case.get(cid),
        )
        surfaces.append(surface)
        summaries.append(summary)

    sweep_surface = pd.concat(surfaces, ignore_index=True)
    sweep_summary = pd.DataFrame(summaries)

    sweep_surface.to_csv(SWEEP_RESULTS_DIR / "surface_pressure_sweep.csv", index=False)
    sweep_summary.to_csv(SWEEP_RESULTS_DIR / "sweep_summary.csv", index=False)
    print(f"Wrote {len(sweep_surface):,} surface rows from {total} cases.")


if __name__ == "__main__":
    main()
