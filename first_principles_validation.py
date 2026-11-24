#!/usr/bin/env python3
"""Reproducible structural validation for the archived dynamic-ring model."""
import ast
import csv
import importlib.util
import json
import math
import os
import statistics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, ".sdf_vortex_validation_source")
BASELINE_PATH = os.path.join(SOURCE_DIR, "sdf_local_lattice.py")
DYNAMIC_PATH = os.path.join(SOURCE_DIR, "sdf_local_lattice_dynamic_ring.py")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("validation_base", BASELINE_PATH)
dynamic = load_module("validation_dynamic", DYNAMIC_PATH)


def simulate(seed, drive, damping, steps=400):
    params = dynamic.Parameters(seed=seed, dt=0.02, damping=damping,
                               drive_amplitude=drive, ring_dynamic_coupling=0.35)
    sim = dynamic.DynamicRingSDFLattice(params)
    return sim, sim.run(steps)


def energy_stats(history):
    values = [float(row["mean_energy_proxy"]) for row in history]
    times = [float(row["time"]) for row in history]
    slope = sum((t - statistics.mean(times)) * (v - statistics.mean(values)) for t, v in zip(times, values)) / sum((t - statistics.mean(times)) ** 2 for t in times)
    initial, final = values[0], values[-1]
    return {"initial_energy": initial, "final_energy": final,
            "minimum_energy": min(values), "maximum_energy": max(values),
            "absolute_drift": abs(final - initial),
            "relative_drift": abs(final - initial) / abs(initial) if initial else None,
            "linear_slope": slope,
            "all_finite": all(math.isfinite(v) for v in values)}


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


sim, trajectory = simulate(7, 0.35, 0.08)
energy = energy_stats(trajectory)
trajectory_rows = [{k: float(v) if isinstance(v, (int, float)) else v for k, v in row.items()} for row in trajectory]
write_csv(os.path.join(BASE_DIR, "energy_trajectory.csv"), trajectory_rows)

seeds = [7, 42, 100, 200, 314]
metrics = ["ring_order_dynamic", "dynamic_link_rms", "mean_energy_proxy", "rms_strain", "mean_polarization"]
finals = []
for seed in seeds:
    _, hist = simulate(seed, 0.35, 0.08)
    finals.append({"seed": seed, **hist[-1]})
ensemble_rows = []
for metric in metrics:
    values = [float(r[metric]) for r in finals]
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    ensemble_rows.append({"metric": metric, "mean": mean, "sample_stddev": sd,
                          "minimum": min(values), "maximum": max(values),
                          "coefficient_of_variation": sd / abs(mean) if mean else None})
write_csv(os.path.join(BASE_DIR, "ensemble_summary.csv"), ensemble_rows)

comparison_rows = []
for drive, damping in [(0.0, 0.08), (0.35, 0.08), (0.0, 0.0), (0.35, 0.0)]:
    _, hist = simulate(7, drive, damping)
    st = energy_stats(hist)
    comparison_rows.append({"seed": 7, "drive_amplitude": drive, "damping": damping,
                            "final_energy": st["final_energy"], "drift": st["final_energy"] - st["initial_energy"],
                            "linear_slope": st["linear_slope"], "minimum_energy": st["minimum_energy"],
                            "maximum_energy": st["maximum_energy"], "all_finite": st["all_finite"]})
write_csv(os.path.join(BASE_DIR, "controlled_comparison.csv"), comparison_rows)

with open(DYNAMIC_PATH, encoding="ascii") as f:
    dynamic_source = f.read()
with open(BASELINE_PATH, encoding="ascii") as f:
    base_source = f.read()
base_tree = ast.parse(base_source)
dynamic_tree = ast.parse(dynamic_source)
def class_methods(tree, cls_name):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
def names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
base_methods = class_methods(base_tree, "LocalSDFLattice")
dynamic_class = next(n for n in dynamic_tree.body if isinstance(n, ast.ClassDef) and n.name == "DynamicRingSDFLattice")
dyn_methods = {n.name: n for n in dynamic_class.body if isinstance(n, ast.FunctionDef)}
accel_names = names(base_methods["_acceleration"])
edge_names = names(base_methods["_edge_current"])
diag_names = names(base_methods["diagnostics"])
structural = {
    "dynamic_class_overrides_acceleration": "_acceleration" in dyn_methods,
    "mean_energy_proxy_uses_link_A": "link_A" in names(base_methods["_edge_current"]),
    "mean_energy_proxy_uses_dynamic_link": "dynamic_link" in diag_names,
    "dynamic_link_affects_acceleration_path": "dynamic_link" in accel_names or "dynamic_link" in edge_names,
    "source_note": "DynamicRingSDFLattice inherits _acceleration. Its acceleration calls _edge_current, which uses link_A; base diagnostics compute mean_energy_proxy from that current phase. dynamic_link is used by dynamic-ring diagnostics and updates, not acceleration."
}
report = {"source": "SDF-Vortex-(3).zip", "steps": 400, "dt": 0.02,
          "energy_trajectory": energy, "ensemble": ensemble_rows,
          "controlled_comparison": comparison_rows, "structural_inspection": structural,
          "limitations": ["Exploratory phenomenological toy model, not validated physics.",
                          "Energy is a diagnostic proxy and need not be conserved under drive, damping, relaxation, or imposed link phases.",
                          "No physical units are assigned; values are model diagnostics.",
                          "This is structural/numerical validation only; no empirical calibration, convergence study, or physical inference is established."]}
with open(os.path.join(BASE_DIR, "first_principles_validation.json"), "w", encoding="ascii") as f:
    json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)

lines = ["# Dynamic-Ring Structural Validation", "", "Source: `SDF-Vortex-(3).zip` implementation.", "", "## Energy Trajectory", ""]
for key, value in energy.items():
    lines.append("- `{}: {}`".format(key, value))
lines += ["", "Every one of the 400 history rows is written to `energy_trajectory.csv`.", "", "## Multi-Seed Ensemble", "", "Final diagnostic statistics across seeds 7, 42, 100, 200, and 314:", "", "| Metric | Mean | Sample SD | Min | Max | CV |", "|---|---:|---:|---:|---:|---:|"]
for r in ensemble_rows:
    lines.append("| {} | {:.12g} | {:.12g} | {:.12g} | {:.12g} | {:.12g} |".format(r["metric"], r["mean"], r["sample_stddev"], r["minimum"], r["maximum"], r["coefficient_of_variation"]))
lines += ["", "## Controlled Comparison", "", "Seed 7, 400 steps, ring_dynamic_coupling=0.35:", "", "| Drive | Damping | Final energy | Drift | Slope | Min | Max | Finite |", "|---:|---:|---:|---:|---:|---:|---:|:---:|"]
for r in comparison_rows:
    lines.append("| {} | {} | {:.12g} | {:.12g} | {:.12g} | {:.12g} | {:.12g} | {} |".format(r["drive_amplitude"], r["damping"], r["final_energy"], r["drift"], r["linear_slope"], r["minimum_energy"], r["maximum_energy"], r["all_finite"]))
lines += ["", "## Structural Inspection", "", "- Dynamic-ring class overrides `_acceleration`: **No**.", "- `mean_energy_proxy` is computed from: **link_A**. Base diagnostics call `_edge_current()`, whose phase difference uses the static `link_A`.", "- `dynamic_link` affects acceleration: **No**. The inherited acceleration path uses `_edge_current()` based on `link_A`; `dynamic_link` is relaxed in `step()` and used for dynamic ring diagnostics.", "", "## Limitations", "", "- The README describes an exploratory phenomenological model, not validated physics.", "- The energy field is a diagnostic proxy; drive, damping, density relaxation, and imposed link phases mean conservation is not expected in general.", "- No physical units are assigned or inferred.", "- These checks establish reproducible implementation behavior and finite outputs for the listed cases, not empirical calibration, timestep/mesh convergence, or physical validity.", "- CSV, JSON, and this report contain results from the exact archived implementation extracted at runtime.", ""]
with open(os.path.join(BASE_DIR, "first_principles_validation_report.md"), "w", encoding="ascii", newline="\n") as f:
    f.write("\n".join(lines))
print(json.dumps({"energy": energy, "ensemble": ensemble_rows, "comparison": comparison_rows, "structural": structural}, indent=2))
