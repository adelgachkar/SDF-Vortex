#!/usr/bin/env python3
"""Self-contained tests for a phenomenological dynamic-ring SDF lattice."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Parameters:
    n_cells: int = 6
    dt: float = 0.02
    inertia: float = 1.0
    damping: float = 0.08
    stiffness: float = 1.0
    ring_flux: float = 0.20
    leak_strength: float = 0.30
    density_coupling: float = 0.18
    density_relaxation: float = 0.8
    strain_coupling: float = 0.20
    polarization_coupling: float = 0.25
    drive_amplitude: float = 0.35
    ring_dynamic_coupling: float = 0.0
    dynamic_link_feedback: bool = True
    seed: int = 7

    def validate(self):
        if not isinstance(self.n_cells, int) or self.n_cells < 2:
            raise ValueError("n_cells must be an integer >= 2")
        for name in ("dt", "inertia", "stiffness", "density_relaxation"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        for name in ("damping", "ring_flux", "leak_strength", "density_coupling",
                     "strain_coupling", "polarization_coupling", "drive_amplitude",
                     "ring_dynamic_coupling"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        return self


class BaseLattice:
    """Fixed-link local lattice with the base SDF phase and field updates."""
    def __init__(self, params):
        self.p = params.validate()
        self.n = 4 * params.n_cells
        rng = np.random.default_rng(params.seed)
        edges, kinds = [], []
        for cell in range(params.n_cells):
            base = 4 * cell
            for a in range(4):
                for b in range(a + 1, 4):
                    edges.append((base + a, base + b))
                    kinds.append("tetra")
        for cell in range(params.n_cells - 1):
            edges.append((4 * cell, 4 * (cell + 1)))
            kinds.append("axial")
        self.edges = np.asarray(edges, dtype=int)
        self.edge_kind = np.asarray(kinds)
        self.phi = rng.normal(0.0, 0.025, self.n)
        self.velocity = np.zeros(self.n)
        self.density = np.zeros(self.n)
        self.polarization = np.zeros((self.n, 3))
        x = np.repeat(np.arange(params.n_cells), 4).astype(float)
        self.leak = params.leak_strength * np.exp(-x / max(1, params.n_cells - 1))
        self.link_A = np.zeros(len(self.edges))
        self.link_A[self.edge_kind == "tetra"] = params.ring_flux / 3.0
        self.dynamic_link = self.link_A.copy()
        self.force = np.zeros(self.n)
        self.force[0] = params.drive_amplitude
        self.time = 0.0
        self.faces = []
        for cell in range(params.n_cells):
            b = 4 * cell
            self.faces.extend(((b, b + 1, b + 2), (b, b + 1, b + 3),
                               (b, b + 2, b + 3), (b + 1, b + 2, b + 3)))
        self.faces = np.asarray(self.faces, dtype=int)
        self._last_acc = self._acceleration()

    def _phase_links(self):
        return self.link_A

    def _edge_current(self):
        i, j = self.edges.T
        phase_difference = self.phi[j] - self.phi[i] - self._phase_links()
        return self.p.stiffness * np.sin(phase_difference), phase_difference

    def _acceleration(self):
        current, _ = self._edge_current()
        net = np.zeros(self.n)
        i, j = self.edges.T
        np.add.at(net, i, current)
        np.add.at(net, j, -current)
        net -= self.p.density_coupling * self.density
        net += self.force
        return (net - self.p.damping * self.velocity) / self.p.inertia

    def _update_fields(self):
        i, j = self.edges.T
        edge_strain = np.zeros(self.n)
        np.add.at(edge_strain, i, self.phi[j] - self.phi[i])
        np.add.at(edge_strain, j, self.phi[i] - self.phi[j])
        target_density = self.p.strain_coupling * edge_strain + self.leak * np.cos(self.phi)
        self.density += self.p.dt * self.p.density_relaxation * (target_density - self.density)
        grad = np.zeros(self.n)
        np.add.at(grad, i, self.phi[j] - self.phi[i])
        np.add.at(grad, j, self.phi[j] - self.phi[i])
        target_p = np.zeros_like(self.polarization)
        target_p[:, 0] = grad + self.p.polarization_coupling * self.density
        self.polarization += self.p.dt * (target_p - self.polarization)
        return edge_strain

    def step(self):
        dt = self.p.dt
        self.phi += dt * self.velocity + 0.5 * dt * dt * self._last_acc
        self.velocity += 0.5 * dt * self._last_acc
        self._update_fields()
        new_acc = self._acceleration()
        self.velocity += 0.5 * dt * new_acc
        self._last_acc = new_acc
        self.time += dt
        arrays = (self.phi, self.velocity, self.density, self.polarization)
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise FloatingPointError("Non-finite state during integration")
        return self.diagnostics()

    def run(self, steps):
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        result = None
        for _ in range(steps):
            result = self.step()
        return result

    def face_fluxes(self, links=None, edges=None):
        if links is None:
            links = self.link_A
        if edges is None:
            edges = self.edges
        edge_map = {tuple(map(int, edge)): k for k, edge in enumerate(edges)}
        fluxes = []
        for a, b, c in self.faces:
            total = 0.0
            for u, v in ((a, b), (b, c), (c, a)):
                if (int(u), int(v)) in edge_map:
                    total += links[edge_map[(int(u), int(v))]]
                else:
                    total -= links[edge_map[(int(v), int(u))]]
            fluxes.append(np.angle(np.exp(1j * total)))
        return np.asarray(fluxes)

    def diagnostics(self):
        _, diff = self._edge_current()
        static_flux = self.face_fluxes(self.link_A)
        dynamic_flux = self.face_fluxes(self.dynamic_link)
        return {
            "time": float(self.time),
            "ring_order_static": float(np.mean(np.cos(static_flux))),
            "ring_order_dynamic": float(np.mean(np.cos(dynamic_flux))),
            "ring_flux_rms": float(np.sqrt(np.mean(dynamic_flux ** 2))),
            "dynamic_link_rms": float(np.sqrt(np.mean(self.dynamic_link ** 2))),
            "mean_energy_proxy": float(0.5 * self.p.inertia * np.mean(self.velocity ** 2)
                                        + self.p.stiffness * np.mean(1.0 - np.cos(diff))),
            "mean_density": float(np.mean(self.density)),
            "mean_polarization": float(np.mean(np.linalg.norm(self.polarization, axis=1))),
        }


class DynamicRingLattice(BaseLattice):
    """Dynamic link relaxation, optionally coupled back into phase acceleration."""
    def _phase_links(self):
        if self.p.dynamic_link_feedback and self.p.ring_dynamic_coupling > 0.0:
            return self.dynamic_link
        return self.link_A

    def step(self):
        dt = self.p.dt
        self.phi += dt * self.velocity + 0.5 * dt * dt * self._last_acc
        self.velocity += 0.5 * dt * self._last_acc
        edge_strain = self._update_fields()
        if self.p.ring_dynamic_coupling > 0.0:
            tetra = self.edge_kind == "tetra"
            mean_strain = float(np.mean(np.abs(edge_strain)))
            mean_pol = float(np.mean(np.linalg.norm(self.polarization, axis=1)))
            shift = self.p.ring_dynamic_coupling * 0.1 * (mean_strain + 0.5 * mean_pol)
            target = self.link_A + tetra * shift
            self.dynamic_link += dt * 0.5 * (target - self.dynamic_link)
        new_acc = self._acceleration()
        self.velocity += 0.5 * dt * new_acc
        self._last_acc = new_acc
        self.time += dt
        arrays = (self.phi, self.velocity, self.density, self.polarization, self.dynamic_link)
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise FloatingPointError("Non-finite state during integration")
        return self.diagnostics()


def assert_close(name, left, right, atol=1e-12):
    if not np.allclose(left, right, rtol=0.0, atol=atol):
        raise AssertionError(name + " check failed")


def run_checks(steps=250):
    checks = {}
    params0 = Parameters(seed=19, drive_amplitude=0.35, ring_dynamic_coupling=0.0)
    base = BaseLattice(params0)
    dynamic0 = DynamicRingLattice(params0)
    for _ in range(steps):
        base.step()
        dynamic0.step()
    assert_close("zero-coupling phase trajectory", base.phi, dynamic0.phi)
    assert_close("zero-coupling velocity trajectory", base.velocity, dynamic0.velocity)
    assert_close("zero-coupling density trajectory", base.density, dynamic0.density)
    assert_close("zero-coupling links", dynamic0.dynamic_link, dynamic0.link_A)
    checks["zero_coupling_matches_baseline"] = True

    active_p = Parameters(seed=19, ring_dynamic_coupling=0.9, dynamic_link_feedback=True)
    active = DynamicRingLattice(active_p)
    control = DynamicRingLattice(Parameters(seed=19, ring_dynamic_coupling=0.9,
                                             dynamic_link_feedback=False))
    active.run(steps)
    control.run(steps)
    feedback_acc = active._acceleration()
    no_feedback_acc = control._acceleration()
    acceleration_change = float(np.max(np.abs(feedback_acc - no_feedback_acc)))
    if not acceleration_change > 1e-10:
        raise AssertionError("dynamic_link did not affect phase acceleration")
    assert float(np.max(np.abs(active.dynamic_link - active.link_A))) > 1e-8
    checks["positive_coupling_changes_phase_acceleration"] = True
    checks["feedback_acceleration_max_difference"] = acceleration_change

    gauge = DynamicRingLattice(active_p)
    gauge.run(steps // 3)
    before_acc = gauge._acceleration().copy()
    before_diag = gauge.diagnostics()
    shift = np.random.default_rng(1234).normal(0.0, 0.4, gauge.n)
    i, j = gauge.edges.T
    gauge.phi += shift
    gauge.link_A += shift[j] - shift[i]
    gauge.dynamic_link += shift[j] - shift[i]
    after_acc = gauge._acceleration()
    after_diag = gauge.diagnostics()
    assert_close("gauge acceleration", before_acc, after_acc, atol=2e-12)
    for key in ("ring_order_static", "ring_order_dynamic", "ring_flux_rms", "mean_energy_proxy"):
        assert_close("gauge " + key, before_diag[key], after_diag[key], atol=2e-12)
    checks["gauge_invariance"] = True

    oriented = DynamicRingLattice(active_p)
    oriented.run(steps // 3)
    original_edges = oriented.edges.copy()
    original_links = oriented.dynamic_link.copy()
    original_differences = (oriented.phi[original_edges[:, 1]]
                            - oriented.phi[original_edges[:, 0]] - original_links)
    original_current = oriented.p.stiffness * np.sin(original_differences)
    original_net = np.zeros(oriented.n)
    np.add.at(original_net, original_edges[:, 0], original_current)
    np.add.at(original_net, original_edges[:, 1], -original_current)
    original_flux = oriented.face_fluxes(original_links, original_edges)
    reversed_edges = original_edges[:, ::-1].copy()
    reversed_links = -original_links
    reverse_diff = (oriented.phi[reversed_edges[:, 1]]
                    - oriented.phi[reversed_edges[:, 0]] - reversed_links)
    reverse_current = oriented.p.stiffness * np.sin(reverse_diff)
    reverse_net = np.zeros(oriented.n)
    np.add.at(reverse_net, reversed_edges[:, 0], reverse_current)
    np.add.at(reverse_net, reversed_edges[:, 1], -reverse_current)
    reverse_flux = oriented.face_fluxes(reversed_links, reversed_edges)
    assert_close("orientation node force", original_net, reverse_net)
    assert_close("orientation face flux", original_flux, reverse_flux)
    assert_close("orientation energy phase", np.cos(original_differences), np.cos(reverse_diff))
    checks["orientation_invariance"] = True

    ensemble = []
    for seed in (7, 42, 100):
        p = Parameters(seed=seed, ring_dynamic_coupling=0.45, dynamic_link_feedback=True)
        sim = DynamicRingLattice(p)
        final = sim.run(steps)
        ensemble.append({"seed": seed,
                         "ring_order": final["ring_order_dynamic"],
                         "energy_proxy": final["mean_energy_proxy"]})
    summary = {}
    for key in ("ring_order", "energy_proxy"):
        values = np.asarray([item[key] for item in ensemble], dtype=float)
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}
    checks["ensemble_seed_count"] = len(ensemble)
    return {
        "model": "self-contained phenomenological SDF local lattice with dynamic links",
        "steps_per_check": steps,
        "parameters": asdict(active_p),
        "checks": checks,
        "ensemble": {"seeds": ensemble, "statistics_population_std": summary},
        "orientation_convention": "edge links are directed; reversing an edge also negates its link",
        "gauge_convention": "phi_i -> phi_i + chi_i; A_ij -> A_ij + chi_j - chi_i",
        "caveat": "Numerical toy-model verification only; not physical validation.",
    }


def main():
    result = run_checks()
    output = Path("/mnt/data/foundation_followup/foundation_followup_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
                      encoding="ascii")
    print("PASS: zero-coupling trajectory matches the fixed-link baseline")
    print("PASS: positive coupling changes phase acceleration through dynamic_link")
    print("PASS: gauge and edge-orientation invariance")
    print("PASS: ensemble seeds 7, 42, 100")
    for metric, stats in result["ensemble"]["statistics_population_std"].items():
        print("ENSEMBLE {}: mean={:.12g}, std={:.12g}".format(
            metric, stats["mean"], stats["std"]))
    print("WROTE {}".format(output))


if __name__ == "__main__":
    main()
