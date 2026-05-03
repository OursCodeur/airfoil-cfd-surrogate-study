# Airfoil CFD-to-Surrogate Study

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://airfoil-cfd-surrogate-study.streamlit.app/)

Streamlit app that starts from the SU2 NACA 0012 quick-start case, expands it
into a dense Mach/AoA CFD sweep, and compares two surface-pressure surrogates:
a deterministic response-surface interpolator and a jump-aware weighted
regressor.

Live app: https://airfoil-cfd-surrogate-study.streamlit.app/

The study is intentionally scoped as a fixed-geometry, 2D operating-condition
experiment. Its purpose is to make solver artifacts, validation boundaries,
speedup caveats, and surrogate failure modes visible rather than to claim
production CFD accuracy.

## What It Shows

- Official SU2 Quick Start input files for the NACA 0012 case.
- A direct Euler flow solve at Mach 0.8 and 1.25 degrees angle of attack.
- A continuous adjoint solve for drag sensitivity.
- A 500-case CFD sweep: 25 Mach values by 20 angles of attack.
- Surface-pressure training data generated from SU2 `surface_flow.csv` outputs.
- A local response-surface surrogate built from all but one CFD case and tested
  on one held-out operating condition.
- A jump-aware Extra Trees regressor trained with estimated jump-location
  features and higher sample weights near the pressure-jump center and
  shoulder regions.
- Side-by-side plots where SU2, the response surface, and the jump-aware
  regressor are overlaid on the same pressure curves.
- Metrics for the held-out pressure curve: MAE, RMSE, and R2.
- Jump-zone metrics that focus only on points close to the strongest detected
  pressure jump on each airfoil surface.
- Pressure-jump detection to expose where interpolation is likely to smooth a
  moving shock-like feature.
- Solver-vs-surrogate timing comparisons, with hardware caveats shown in the
  app.
- Interactive Mach/AoA exploration with interpolation and extrapolation signals.
- ParaView-like pressure and Mach contour views parsed directly from `flow.vtu`.
- Compact solver artifacts: config, mesh, residual history, logs, and adjoint
  sensitivity.

## Scope and Limits

- Fixed NACA 0012 geometry; the design variables are Mach number and angle of
  attack only.
- CFD data comes from an SU2 Euler quick-start setup and a generated sweep, not
  from a validation-grade industrial CFD campaign.
- The surrogate target is pressure coefficient sampled on a 2D airfoil surface,
  not a 3D geometry, CAD/B-rep, mesh-native model, or multiphysics field.
- The useful output is the workflow: solver data generation, holdout validation,
  interpolation-risk analysis, moving pressure-jump detection, local loss
  weighting, timing caveats, and trust signals shown beside speedup.

## Requirements

- Python 3.11 or newer.
- No local SU2 install is required to view the bundled results or build the
  surrogate inside the app.
- SU2 is only required to regenerate the CFD outputs locally.

## Quick Start

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Open the URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Rerunning SU2

The bundled results were generated with SU2 v8.5.0 using the official Linux
OpenMP binary release.

To rerun the single reference case when `SU2_CFD` is available on `PATH`:

```bash
scripts/run_su2.sh all
```

To regenerate the sweep:

```bash
.venv/bin/python scripts/generate_sweep.py --su2-cfd SU2_CFD --threads 16
```

To reuse already generated per-case outputs and run only missing cases:

```bash
.venv/bin/python scripts/generate_sweep.py --su2-cfd SU2_CFD --threads 16 --reuse-existing
```

This writes generated data under:

```text
data/results/sweep/
├── surface_pressure_sweep.csv
└── sweep_summary.csv
```

The sweep summary includes local wall-clock timing per SU2 case. Those timings
are useful for order-of-magnitude comparison with surrogate inference, but they
are not same-hardware benchmarks if the app is later run on another machine or
on Streamlit Community Cloud.

The dense grid reduces the distance between neighboring pressure curves in the
transonic region where the dominant surface-pressure jump moves quickly with
Mach and angle of attack. That region is where pointwise interpolation tends to
smear a sharp pressure drop, so the app visualizes the detected jump location as
a trust signal.
When a generated case is selected for validation, that exact case is withheld
from both holdout surrogates. The design-explorer tab trains on the full sweep,
so an exact generated condition can be much closer there. The app calls out that
difference explicitly.

The jump-aware regressor is trained on numeric features derived from Mach,
angle of attack, surface side, airfoil surface position, and the estimated
pressure-jump location for that operating condition. For each CFD case and
surface, the app detects the strongest interior pressure jump, then assigns
higher sample weight to the jump center and to upstream/downstream shoulder
points around the break. This keeps global curve quality visible while
demonstrating that a physically important local feature may deserve a different
loss weighting than smooth regions.

## Project Structure

```text
.
├── app.py
├── data/
│   ├── input/
│   │   ├── inv_NACA0012.cfg
│   │   ├── inv_NACA0012_adjoint.cfg
│   │   └── mesh_NACA0012_inv.su2
│   └── results/
│       ├── adjoint/
│       │   ├── adjoint.log
│       │   ├── history_adjoint.csv
│       │   └── surface_adjoint.csv
│       ├── direct/
│       │   ├── direct.log
│       │   ├── flow.vtu
│       │   ├── history_direct.csv
│       │   └── surface_flow.csv
│       └── sweep/
│           ├── surface_pressure_sweep.csv
│           └── sweep_summary.csv
├── scripts/
│   ├── generate_sweep.py
│   └── run_su2.sh
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Development Checks

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m compileall app.py scripts/generate_sweep.py
.venv/bin/ruff format --check .
.venv/bin/ruff check .
```

Optional Streamlit smoke test:

```bash
.venv/bin/python -c "from streamlit.testing.v1 import AppTest; at = AppTest.from_file('app.py').run(timeout=120); print(len(at.exception))"
```

## Deployment

The app is ready for Streamlit Community Cloud:

- Repository: any public GitHub repository containing these files.
- Branch: `main`.
- Entrypoint file: `app.py`.
- Python version: select Python 3.12 or newer in Streamlit's advanced settings.
- Secrets: none.

Streamlit Community Cloud reads `requirements.txt` from the repository root and
runs `streamlit run app.py`. The bundled sweep data lets the deployed app run
without SU2 installed.

## Sources

- SU2 Quick Start: https://su2code.github.io/docs_v7/Quick-Start/
- SU2 repository: https://github.com/su2code/SU2
- SU2 release used for the bundled run: https://github.com/su2code/SU2/releases/tag/v8.5.0
