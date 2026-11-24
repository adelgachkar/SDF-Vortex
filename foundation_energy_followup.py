#!/usr/bin/env python3
"""Diagnostic-proxy, dynamic-ring ensemble, and parameter sensitivity follow-up."""
import ast
import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

SOURCE_DIR = Path("/mnt/data/.sdf_vortex_validation_source")
BASE_PATH = SOURCE_DIR / "sdf_local_lattice.py"
DYNAMIC_PATH = SOURCE_DIR / "sdf_local_lattice_dynamic_ring.py"
OUT_DIR = Path("/mnt/data/foundation_followup")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def finite_stats(values):
    arr = np.asarray(values, dtype=float)
    finite = bool(np.all(np.isfinite(arr)))
    return arr, finite


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty CSV: " + str(path))
    with path.open("w", newline="", encoding="ascii") as f:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def slope(times, values):
    return float(np.polyfit(np.asarray(times, dtype=float), np.asarray(values, dtype=float), 1)[0])


def trajectory_summary(rows):
    values = np.asarray([r["mean_energy_proxy"] for r in rows], dtype=float)
    times = np.asarray([r["time"] for r in rows], dtype=float)
    finite = bool(np.all(np.isfinite(values)))
    initial, final = float(values[0]), float(values[-1])
    denom = max(abs(initial), float(np.finfo(float).tiny))
    increments = np.diff(values)
    return {
        "finite": finite, "finite_count": int(np.isfinite(values).sum()),
        "total_count": int(values.size), "initial": initial, "final": final,
        "min": float(np.min(values)), "max": float(np.max(values)),
        "absolute_drift": final - initial,
        "relative_drift": (final - initial) / denom,
        "max_absolute_one_step_increment": float(np.max(np.abs(increments))) if increments.size else 0.0,
        "linear_least_squares_slope": slope(times, values),
        "monotonic_increasing": bool(np.all(increments >= 0)),
        "monotonic_decreasing": bool(np.all(increments <= 0)),
        "first_five": [float(x) for x in values[:5]],
        "last_five": [float(x) for x in values[-5:]],
    }


def source_checks():
    tree = ast.parse(DYNAMIC_PATH.read_text(encoding="ascii"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DynamicRingSDFLattice")
    overrides = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "_acceleration" for n in cls.body)
    base_text = BASE_PATH.read_text(encoding="ascii")
    dyn_text = DYNAMIC_PATH.read_text(encoding="ascii")
    base_tree = ast.parse(base_text)
    diag = next(n for n in ast.walk(base_tree) if isinstance(n, ast.FunctionDef) and n.name == "diagnostics")
    energy_node = next(n for n in ast.walk(diag) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "energy" for t in n.targets))
    energy_expr = ast.unparse(energy_node.value)
    energy_uses_link_A = "link_A" in energy_expr
    energy_uses_dynamic_link = "dynamic_link" in energy_expr
    acceleration = next(n for n in ast.walk(base_tree) if isinstance(n, ast.FunctionDef) and n.name == "_acceleration")
    accel_source = ast.unparse(acceleration)
    dynamic_link_in_acceleration_path = "dynamic_link" in accel_source or ("_acceleration" in dyn_text and "dynamic_link" in dyn_text.split("def _acceleration", 1)[-1].split("def ", 1)[0])
    return {
        "DynamicRingSDFLattice_overrides__acceleration": overrides,
        "mean_energy_proxy_expression": energy_expr,
        "mean_energy_proxy_uses_link_A": energy_uses_link_A,
        "mean_energy_proxy_uses_dynamic_link": energy_uses_dynamic_link,
        "dynamic_link_passed_into_acceleration_path": dynamic_link_in_acceleration_path,
        "note": "Dynamic extension inherits acceleration; no dynamic_link dependency is present in the base acceleration implementation."
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_module("followup_base_sdf", BASE_PATH)
    dyn = load_module("followup_dynamic_sdf", DYNAMIC_PATH)
    common = dict(n_cells=12, dt=0.02, seed=7, drive_amplitude=0.35, ring_dynamic_coupling=0.35)

    params = dyn.Parameters(**common)
    sim = dyn.DynamicRingSDFLattice(params)
    trajectory = sim.run(400)
    trajectory_rows = [{"step": i + 1, **{k: v for k, v in row.items()}} for i, row in enumerate(trajectory)]
    write_csv(OUT_DIR / "energy_proxy_trajectory.csv", trajectory_rows)
    energy_summary = trajectory_summary(trajectory)

    ensemble_rows = []
    final_ring_values = []
    for seed in [7, 42, 100, 200, 314]:
        p = dyn.Parameters(**{**common, "seed": seed})
        instance = dyn.DynamicRingSDFLattice(p)
        hist = instance.run(400)
        value = float(hist[-1]["ring_order_dynamic"])
        finite = bool(all(np.isfinite(v) for row in hist for v in row.values()))
        ensemble_rows.append({"seed": seed, "final_ring_order_dynamic": value, "finite": finite})
        final_ring_values.append(value)
    vals = np.asarray(final_ring_values, dtype=float)
    ensemble_summary = {
        "mean": float(np.mean(vals)), "sample_standard_deviation_ddof_1": float(np.std(vals, ddof=1)),
        "min": float(np.min(vals)), "max": float(np.max(vals)),
        "coefficient_of_variation": float(np.std(vals, ddof=1) / abs(np.mean(vals))) if np.mean(vals) != 0 else None,
    }
    write_csv(OUT_DIR / "ring_order_ensemble.csv", ensemble_rows)

    cases = [
        ("baseline", 0.8, 0.25),
        ("density_relaxation_zero", 0.0, 0.25),
        ("density_relaxation_high", 1.6, 0.25),
        ("polarization_coupling_zero", 0.8, 0.0),
        ("polarization_coupling_high", 0.8, 0.5),
    ]
    comparison = []
    raw_case_results = {}
    for name, relaxation, polarization in cases:
        record = {"case": name, "density_relaxation": relaxation, "polarization_coupling": polarization}
        try:
            p = dyn.Parameters(**{**common, "density_relaxation": relaxation, "polarization_coupling": polarization})
            instance = dyn.DynamicRingSDFLattice(p)
            hist = instance.run(400)
            summary = trajectory_summary(hist)
            finalrow = hist[-1]
            record.update({
                "status": "ok", "trajectory_finite": summary["finite"],
                "initial_mean_energy_proxy": summary["initial"], "final_mean_energy_proxy": summary["final"],
                "min_mean_energy_proxy": summary["min"], "max_mean_energy_proxy": summary["max"],
                "absolute_drift": summary["absolute_drift"], "slope": summary["linear_least_squares_slope"],
                "max_absolute_one_step_increment": summary["max_absolute_one_step_increment"],
                "final_mean_density": float(finalrow["mean_density"]),
                "final_mean_polarization": float(finalrow["mean_polarization"]),
                "final_rms_strain": float(finalrow["rms_strain"]),
                "final_ring_order_dynamic": float(finalrow["ring_order_dynamic"]),
            })
            raw_case_results[name] = record
        except Exception as exc:
            record.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc), "trajectory_finite": None})
            raw_case_results[name] = record
        comparison.append(record)
    baseline = raw_case_results.get("baseline", {})
    for record in comparison:
        if record.get("status") == "ok" and baseline.get("status") == "ok":
            record["final_energy_delta_vs_baseline"] = record["final_mean_energy_proxy"] - baseline["final_mean_energy_proxy"]
            record["drift_delta_vs_baseline"] = record["absolute_drift"] - baseline["absolute_drift"]
        else:
            record["final_energy_delta_vs_baseline"] = None
            record["drift_delta_vs_baseline"] = None
    write_csv(OUT_DIR / "relaxation_polarization_comparison.csv", comparison)

    checks = source_checks()
    self_test = subprocess.run([sys.executable, str(BASE_PATH), "--self-test"], cwd=str(SOURCE_DIR), capture_output=True, text=True)
    results = {
        "analysis_label": "diagnostic-proxy behavior; not a physical energy conservation test",
        "configuration": {"n_cells": 12, "dt": 0.02, "steps": 400, "drive_amplitude": 0.35, "ring_dynamic_coupling": 0.35, "base_seed": 7},
        "energy_proxy_trajectory": energy_summary,
        "ring_order_ensemble": {"seed_results": ensemble_rows, **ensemble_summary},
        "relaxation_polarization_cases": comparison,
        "source_structural_checks": checks,
        "self_test": {"command": f"{sys.executable} {BASE_PATH} --self-test", "cwd": str(SOURCE_DIR), "exit_code": self_test.returncode, "stdout": self_test.stdout, "stderr": self_test.stderr},
        "limitations": ["mean_energy_proxy is not calibrated physical energy.", "Drive, damping, relaxation, and feedback can change the proxy.", "Parameter variations are sensitivity/behavior tests and do not by themselves prove a thermodynamic dissipation law.", "The dynamic_link field is not in acceleration if confirmed by source checks."],
    }
    (OUT_DIR / "foundation_followup_results.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="ascii")

    lines = ["# Foundation Energy Follow-up", "", "All results use the authoritative source modules in `/mnt/data/.sdf_vortex_validation_source`. The energy analysis is diagnostic-proxy behavior, not a physical energy conservation test.", "", "## A. Energy-proxy trajectory", "", "Configuration: n_cells=12, seed=7, dt=0.02, steps=400, drive_amplitude=0.35, ring_dynamic_coupling=0.35; remaining defaults.", "", "```", json.dumps(energy_summary, indent=2, sort_keys=True), "```", "", "## B. Ring-order ensemble", "", "| Seed | Final ring_order_dynamic | Finite |", "|---:|---:|:---:|"]
    for r in ensemble_rows:
        lines.append(f"| {r['seed']} | {r['final_ring_order_dynamic']:.17g} | {r['finite']} |")
    lines += ["", "Ensemble summary (sample standard deviation uses ddof=1):", "", "```", json.dumps(ensemble_summary, indent=2, sort_keys=True), "```", "", "## C. Relaxation and polarization sensitivity", "", "These parameter variations are sensitivity/behavior tests and do not by themselves prove a thermodynamic dissipation law.", "", "| Case | Status | Final energy | Drift | Delta final energy | Delta drift | Finite |", "|---|---|---:|---:|---:|---:|:---:|"]
    for r in comparison:
        if r["status"] == "ok":
            lines.append(f"| {r['case']} | ok | {r['final_mean_energy_proxy']:.17g} | {r['absolute_drift']:.17g} | {r['final_energy_delta_vs_baseline']:.17g} | {r['drift_delta_vs_baseline']:.17g} | {r['trajectory_finite']} |")
        else:
            lines.append(f"| {r['case']} | error: {r['error_type']}: {r['error']} | | | | | |")
    lines += ["", "Full case metrics are in the comparison CSV and JSON.", "", "## D. Source checks", "", "```", json.dumps(checks, indent=2, sort_keys=True), "```", "", "## Self-test", "", f"Command: `{sys.executable} {BASE_PATH} --self-test`", f"Working directory: `{SOURCE_DIR}`", f"Exit code: {self_test.returncode}", "", "Stdout:", "", "```", self_test.stdout.rstrip(), "```", "", "Stderr:", "", "```", self_test.stderr.rstrip(), "```", "", "## Limitations", "", "- The proxy is not calibrated physical energy.", "- Drive, damping, relaxation, and feedback can change it.", "- Parameter variations are sensitivity/behavior tests and do not by themselves prove a thermodynamic dissipation law.", "- Dynamic_link is not in the acceleration path if confirmed by the structural checks above.", ""]
    (OUT_DIR / "foundation_followup_report.md").write_text("\n".join(lines), encoding="ascii")
    print(json.dumps({"output_dir": str(OUT_DIR), "energy": energy_summary, "ensemble": {"rows": ensemble_rows, **ensemble_summary}, "cases": comparison, "source_checks": checks, "self_test_exit_code": self_test.returncode}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
