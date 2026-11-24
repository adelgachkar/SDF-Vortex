# Dynamic-Ring Structural Validation

Source: `SDF-Vortex-(3).zip` implementation.

## Energy Trajectory

- `initial_energy: 0.002273807616352104`
- `final_energy: 0.011361622432979949`
- `minimum_energy: 0.002273807616352104`
- `maximum_energy: 0.011361622432979949`
- `absolute_drift: 0.009087814816627845`
- `relative_drift: 3.9967386648161254`
- `linear_slope: 0.001156541622027061`
- `all_finite: True`

Every one of the 400 history rows is written to `energy_trajectory.csv`.

## Multi-Seed Ensemble

Final diagnostic statistics across seeds 7, 42, 100, 200, and 314:

| Metric | Mean | Sample SD | Min | Max | CV |
|---|---:|---:|---:|---:|---:|
| ring_order_dynamic | 0.997174971445 | 6.23019582163e-06 | 0.997169475057 | 0.99718452674 | 6.24784616545e-06 |
| dynamic_link_rms | 0.0700254170258 | 7.72802560703e-05 | 0.0699068690097 | 0.070093571026 | 0.00110360293951 |
| mean_energy_proxy | 0.0114154970086 | 0.000173593650406 | 0.0112256601834 | 0.0116188668001 | 0.0152068412156 |
| rms_strain | 0.0687244360158 | 0.00231110803704 | 0.0663587761368 | 0.0713207832034 | 0.0336286213612 |
| mean_polarization | 0.239238727837 | 0.00247569011508 | 0.236233817818 | 0.242354930435 | 0.0103481996308 |

## Controlled Comparison

Seed 7, 400 steps, ring_dynamic_coupling=0.35:

| Drive | Damping | Final energy | Drift | Slope | Min | Max | Finite |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.0 | 0.08 | 0.017002538885 | 0.0147287578023 | 0.00192841595784 | 0.00227378108268 | 0.017002538885 | True |
| 0.35 | 0.08 | 0.011361622433 | 0.00908781481663 | 0.00115654162203 | 0.00227380761635 | 0.011361622433 | True |
| 0.0 | 0.0 | 0.0285114278902 | 0.0262376424029 | 0.00329012556093 | 0.00227378548735 | 0.0285114278902 | True |
| 0.35 | 0.0 | 0.0183541836789 | 0.0160803717934 | 0.00199310319528 | 0.00227381188548 | 0.0183541836789 | True |

## Structural Inspection

- Dynamic-ring class overrides `_acceleration`: **No**.
- `mean_energy_proxy` is computed from: **link_A**. Base diagnostics call `_edge_current()`, whose phase difference uses the static `link_A`.
- `dynamic_link` affects acceleration: **No**. The inherited acceleration path uses `_edge_current()` based on `link_A`; `dynamic_link` is relaxed in `step()` and used for dynamic ring diagnostics.

## Limitations

- The README describes an exploratory phenomenological model, not validated physics.
- The energy field is a diagnostic proxy; drive, damping, density relaxation, and imposed link phases mean conservation is not expected in general.
- No physical units are assigned or inferred.
- These checks establish reproducible implementation behavior and finite outputs for the listed cases, not empirical calibration, timestep/mesh convergence, or physical validity.
- CSV, JSON, and this report contain results from the exact archived implementation extracted at runtime.
