#!/usr/bin/env python3
"""Run reproducible dynamic-ring checks for the local SDF lattice toy model."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/data")
SOURCE = ROOT / ".sdf_vortex_validation_source" / "sdf_local_lattice_dynamic_ring.py"
OUT = ROOT / "foundation_followup"
SEEDS = (7, 42, 100)
STEPS = 400


def load_model():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Dynamic model source not found: {SOURCE}")
    spec = importlib.util.spec_from_file_location("foundation_dynamic_model", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def face_flux(sim, links, faces=None):
    edge_index = {tuple(edge): index for index, edge in enumerate(sim.edges)}
    selected_faces = sim._faces if faces is None else faces
    flux = []
    for a, b, c in selected_faces:
        total = 0.0
        for u, v in ((a, b), (b, c), (c, a)):
            if u < v:
                total += links[edge_index[(u, v)]]
            else:
                total -= links[edge_index[(v, u)]]
        flux.append(np.angle(np.exp(1j * total)))
    return np.asarray(flux, dtype=float)


def order_metrics(flux):
    return float(np.mean(np.cos(flux))), float(np.sqrt(np.mean(flux ** 2)))


def stats(values):
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    return {"mean": mean, "sample_std_ddof_1": std,
            "coefficient_of_variation": std / abs(mean) if mean else None}


def main():
    model = load_model()
    OUT.mkdir(parents=True, exist_ok=True)
    common = dict(n_cells=12, dt=0.02, drive_amplitude=0.35)

    coupling_cases = {}
    for coupling in (0.0, 0.35):
        sim = model.DynamicRingSDFLattice(model.Parameters(**common, ring_dynamic_coupling=coupling, seed=7))
        history = sim.run(STEPS)
        coupling_cases[str(coupling)] = {
            "dynamic_link_rms": float(history[-1]["dynamic_link_rms"]),
            "ring_order_dynamic": float(history[-1]["ring_order_dynamic"]),
            "final_acceleration_l2": float(np.linalg.norm(sim._acceleration())),
        }
    zero = model.DynamicRingSDFLattice(model.Parameters(**common, ring_dynamic_coupling=0.0, seed=7))
    active = model.DynamicRingSDFLattice(model.Parameters(**common, ring_dynamic_coupling=0.35, seed=7))
    zero.run(STEPS)
    active.run(STEPS)
    same_state_acc = np.allclose(zero._acceleration(), active._acceleration(), rtol=0.0, atol=1e-14)
    link_probe = active.dynamic_link.copy()
    active.dynamic_link += 0.123
    dynamic_link_affects_acceleration = not np.allclose(active._acceleration(), zero._acceleration(), rtol=0.0, atol=1e-14)
    active.dynamic_link[:] = link_probe
    coupling_check = {
        "cases": coupling_cases,
        "positive_coupling_changes_dynamic_link_from_zero_case": not np.isclose(coupling_cases["0.0"]["dynamic_link_rms"], coupling_cases["0.35"]["dynamic_link_rms"], atol=1e-14),
        "acceleration_same_across_runs_with_identical_physical_state": bool(same_state_acc),
        "dynamic_link_directly_affects_acceleration": bool(dynamic_link_affects_acceleration),
        "interpretation": "The extension evolves dynamic_link and diagnoses it, but inherited acceleration uses static link_A; changing only dynamic_link does not change acceleration."
    }
    assert coupling_check["positive_coupling_changes_dynamic_link_from_zero_case"]
    assert not coupling_check["dynamic_link_directly_affects_acceleration"]

    ensemble_rows = []
    for seed in SEEDS:
        sim = model.DynamicRingSDFLattice(model.Parameters(**common, ring_dynamic_coupling=0.35, seed=seed))
        final = sim.run(STEPS)[-1]
        ensemble_rows.append({"seed": seed, "final_ring_order": float(final["ring_order_dynamic"]),
                              "final_energy_proxy": float(final["mean_energy_proxy"]),
                              "dynamic_link_rms": float(final["dynamic_link_rms"]), "steps": STEPS})
    ensemble = {
        "seeds": list(SEEDS), "steps": STEPS,
        "ring_order": stats([r["final_ring_order"] for r in ensemble_rows]),
        "energy_proxy": stats([r["final_energy_proxy"] for r in ensemble_rows]),
        "seed_results": ensemble_rows,
    }

    gauge = model.DynamicRingSDFLattice(model.Parameters(**common, ring_dynamic_coupling=0.35, seed=42))
    gauge.run(STEPS)
    original_phi = gauge.phi.copy()
    before = gauge.diagnostics()
    gauge.phi += 1.23456789
    after = gauge.diagnostics()
    gauge_order_delta = abs(before["ring_order_dynamic"] - after["ring_order_dynamic"])
    gauge_rms_delta = abs(before["rms_strain"] - after["rms_strain"])
    gauge_energy_delta = abs(before["mean_energy_proxy"] - after["mean_energy_proxy"])
    gauge.phi[:] = original_phi

    original_faces = gauge._faces.copy()
    forward_flux = face_flux(gauge, gauge.dynamic_link, original_faces)
    reversed_faces = original_faces[:, ::-1].copy()
    reverse_flux = face_flux(gauge, gauge.dynamic_link, reversed_faces)
    orient_before = order_metrics(forward_flux)
    orient_after = order_metrics(reverse_flux)
    orientation_order_delta = abs(orient_before[0] - orient_after[0])
    orientation_rms_delta = abs(orient_before[1] - orient_after[1])
    invariance = {
        "global_phase_shift": 1.23456789,
        "global_phase_ring_order_abs_delta": gauge_order_delta,
        "global_phase_rms_strain_abs_delta": gauge_rms_delta,
        "global_phase_energy_proxy_abs_delta": gauge_energy_delta,
        "orientation_reversal_ring_order_abs_delta": orientation_order_delta,
        "orientation_reversal_flux_rms_abs_delta": orientation_rms_delta,
        "tolerance": 1e-12,
    }
    assert max(gauge_order_delta, gauge_rms_delta, gauge_energy_delta, orientation_order_delta, orientation_rms_delta) < 1e-12

    results = {
        "model_source": str(SOURCE),
        "model_status": "exploratory phenomenological toy model; not validated physics",
        "configuration": {"n_cells": 12, "dt": 0.02, "steps": STEPS, "drive_amplitude": 0.35, "ring_dynamic_coupling": 0.35},
        "coupling_and_acceleration_check": coupling_check,
        "ensemble": ensemble,
        "invariance_checks": invariance,
        "all_assertions_passed": True,
    }
    json_path = OUT / "foundation_followup_results.json"
    csv_path = OUT / "ring_order_ensemble.csv"
    report_path = OUT / "foundation_followup_report.md"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="ascii")
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ensemble_rows[0]))
        writer.writeheader()
        writer.writerows(ensemble_rows)

    lines = ["# Foundation Follow-up", "", "This report uses the local dynamic-ring extension and a 400-step run. The model is an exploratory phenomenological toy model, not validated physics.", "", "## Coupling and Acceleration", "", "`ring_dynamic_coupling=0` leaves the dynamic link at its static initialization; positive coupling evolves it. The acceleration implementation inherits the base model and uses `link_A`, not `dynamic_link`. Directly perturbing only `dynamic_link` therefore does not affect acceleration.", "", f"- Link RMS at zero coupling: {coupling_cases['0.0']['dynamic_link_rms']:.17g}", f"- Link RMS at coupling 0.35: {coupling_cases['0.35']['dynamic_link_rms']:.17g}", f"- Dynamic link affects acceleration: {dynamic_link_affects_acceleration}", "", "## Multi-Seed Ensemble", "", "Sample standard deviation uses ddof=1; CV is sample standard deviation divided by absolute mean.", "", "| Seed | Final ring order | Final energy proxy | Dynamic-link RMS |", "|---:|---:|---:|---:|"]
    for row in ensemble_rows:
        lines.append(f"| {row['seed']} | {row['final_ring_order']:.17g} | {row['final_energy_proxy']:.17g} | {row['dynamic_link_rms']:.17g} |")
    lines += ["", f"Ring order: mean={ensemble['ring_order']['mean']:.17g}, std={ensemble['ring_order']['sample_std_ddof_1']:.17g}, CV={ensemble['ring_order']['coefficient_of_variation']:.17g}.", f"Energy proxy: mean={ensemble['energy_proxy']['mean']:.17g}, std={ensemble['energy_proxy']['sample_std_ddof_1']:.17g}, CV={ensemble['energy_proxy']['coefficient_of_variation']:.17g}.", "", "## Orientation and Gauge Checks", "", f"- Global phase shift ring-order delta: {gauge_order_delta:.3g}", f"- Global phase shift RMS-strain delta: {gauge_rms_delta:.3g}", f"- Global phase shift energy-proxy delta: {gauge_energy_delta:.3g}", f"- Face-orientation reversal cosine-order delta: {orientation_order_delta:.3g}", f"- Face-orientation reversal flux-RMS delta: {orientation_rms_delta:.3g}", "", "All invariance checks passed at absolute tolerance 1e-12. A global phase shift preserves edge differences; reversing face orientation flips flux signs while preserving cosine order and RMS.", "", "## Outputs", "", f"- JSON: `{json_path}`", f"- CSV: `{csv_path}`", f"- Report: `{report_path}`", ""]
    report_path.write_text("\n".join(lines), encoding="ascii")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "report": str(report_path), "coupling_check": coupling_check, "ensemble": ensemble, "invariance_checks": invariance}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
