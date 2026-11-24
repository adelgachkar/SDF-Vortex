#!/usr/bin/env python3
"""Standalone regression harness for the SDF local lattice and dynamic ring.

The baseline equations and dynamic-link relaxation follow the source in
SDF-Vortex-(3).zip. This file has no dependency on the archive at runtime.

Edge-current architecture:
  static (default): link_eff = link_A. This preserves baseline compatibility.
  dynamic:          link_eff = dynamic_link. The evolving ring field feeds the
                    phase force/current, not only ring diagnostics.
  blend:            link_eff = (1 - blend) * link_A + blend * dynamic_link,
                    with blend in [0, 1].
The separate ring_dynamic_coupling controls evolution of dynamic_link; it is
not silently substituted for the blend fraction. mean_energy_proxy is a
 diagnostic only; no conservation claim is made for this model.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import unittest
import numpy as np


@dataclass(frozen=True)
class Parameters:
    n_cells: int = 12
    dt: float = 0.02
    inertia: float = 1.0
    damping: float = 0.08
    stiffness: float = 1.0
    link_phase: float = 0.25
    ring_flux: float = 0.20
    leak_strength: float = 0.30
    density_coupling: float = 0.18
    density_relaxation: float = 0.8
    strain_coupling: float = 0.20
    polarization_coupling: float = 0.25
    seed: int = 7
    drive_amplitude: float = 0.35
    ring_dynamic_coupling: float = 0.0

    def validate(self):
        if not isinstance(self.n_cells, int) or self.n_cells < 2:
            raise ValueError('n_cells must be an integer >= 2')
        for name in ('dt', 'inertia', 'stiffness', 'density_relaxation'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be finite and positive')
        for name in ('damping', 'link_phase', 'ring_flux', 'leak_strength',
                     'density_coupling', 'strain_coupling', 'polarization_coupling',
                     'drive_amplitude', 'ring_dynamic_coupling'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(name + ' must be finite and nonnegative')
        return self


class LocalSDFLattice:
    """Standalone copy of the baseline lattice equations from the ZIP."""
    def __init__(self, params=Parameters()):
        self.p = params.validate()
        self.n = 4 * params.n_cells
        rng = np.random.default_rng(params.seed)
        edges = []
        kinds = []
        for c in range(params.n_cells):
            base = 4 * c
            for a in range(4):
                for b in range(a + 1, 4):
                    edges.append((base + a, base + b))
                    kinds.append('tetra')
        for c in range(params.n_cells - 1):
            edges.append((4 * c, 4 * (c + 1)))
            kinds.append('axial')
        self.edges = np.asarray(edges, dtype=int)
        self.edge_kind = np.asarray(kinds)
        self.phi = rng.normal(0, 0.025, self.n)
        self.velocity = np.zeros(self.n)
        self.density = np.zeros(self.n)
        self.polarization = np.zeros((self.n, 3))
        x = np.repeat(np.arange(params.n_cells), 4).astype(float)
        self.leak = params.leak_strength * np.exp(-x / max(1, params.n_cells - 1))
        self.link_A = np.zeros(len(self.edges))
        self.link_A[self.edge_kind == 'tetra'] = params.ring_flux / 3.0
        self.force = np.zeros(self.n)
        self.force[0] = 0.35
        self.time = 0.0
        faces = []
        for c in range(params.n_cells):
            b = 4 * c
            faces.extend([(b, b + 1, b + 2), (b, b + 1, b + 3),
                          (b, b + 2, b + 3), (b + 1, b + 2, b + 3)])
        self._faces = np.asarray(faces, dtype=int)
        self._last_acc = self._acceleration()

    def _current_link(self):
        return self.link_A

    def _edge_current(self):
        i, j = self.edges.T
        d = self.phi[j] - self.phi[i] - self._current_link()
        return self.p.stiffness * np.sin(d), d

    def _acceleration(self):
        cur, _ = self._edge_current()
        net = np.zeros(self.n)
        i, j = self.edges.T
        np.add.at(net, i, cur)
        np.add.at(net, j, -cur)
        net -= self.p.density_coupling * self.density
        net += self.force
        return (net - self.p.damping * self.velocity) / self.p.inertia

    def step(self):
        dt = self.p.dt
        a = self._last_acc
        self.phi += dt * self.velocity + 0.5 * dt * dt * a
        self.velocity += 0.5 * dt * a
        i, j = self.edges.T
        edge_strain = np.zeros(self.n)
        np.add.at(edge_strain, i, self.phi[j] - self.phi[i])
        np.add.at(edge_strain, j, self.phi[i] - self.phi[j])
        target = self.p.strain_coupling * edge_strain + self.leak * np.cos(self.phi)
        self.density += dt * self.p.density_relaxation * (target - self.density)
        grad = np.zeros(self.n)
        np.add.at(grad, i, self.phi[j] - self.phi[i])
        np.add.at(grad, j, self.phi[j] - self.phi[i])
        target_p = np.zeros_like(self.polarization)
        target_p[:, 0] = grad + self.p.polarization_coupling * self.density
        self.polarization += dt * (target_p - self.polarization)
        anew = self._acceleration()
        self.velocity += 0.5 * dt * anew
        self._last_acc = anew
        self.time += dt
        if not all(np.all(np.isfinite(v)) for v in
                   (self.phi, self.velocity, self.density, self.polarization)):
            raise FloatingPointError('non-finite state; reduce dt or coupling strengths')
        return self.diagnostics()

    def diagnostics(self):
        _, d = self._edge_current()
        flux = face_fluxes(self.link_A, self.edges, self._faces)
        strain = float(np.sqrt(np.mean(d * d)))
        rho = float(np.mean(self.density))
        pol = float(np.mean(np.linalg.norm(self.polarization, axis=1)))
        order = float(np.mean(np.cos(flux)))
        # This proxy is diagnostic only and is not asserted to be conserved.
        energy_proxy = float(0.5 * self.p.inertia * np.mean(self.velocity ** 2) +
                             self.p.stiffness * np.mean(1 - np.cos(d)))
        return {'time': self.time, 'mean_density': rho, 'rms_strain': strain,
                'mean_polarization': pol, 'ring_order': order,
                'mean_energy_proxy': energy_proxy,
                'max_abs_phase': float(np.max(np.abs(self.phi)))}

    def run(self, steps):
        if not isinstance(steps, int) or steps < 1:
            raise ValueError('steps must be >= 1')
        return [self.step() for _ in range(steps)]


def face_fluxes(links, edges, faces):
    """Return wrapped oriented face fluxes using canonical edge orientation."""
    edge_index = {tuple(edge): k for k, edge in enumerate(edges)}
    out = []
    for a, b, c in faces:
        total = 0.0
        for u, v in ((a, b), (b, c), (c, a)):
            if u < v:
                total += links[edge_index[(u, v)]]
            else:
                total -= links[edge_index[(v, u)]]
        out.append(np.angle(np.exp(1j * total)))
    return np.asarray(out, dtype=float)


class DynamicRingSDFLattice(LocalSDFLattice):
    """Dynamic ring extension with explicit current-link selection."""
    def __init__(self, params=None, edge_current_mode='static', blend=0.5):
        if params is None:
            params = Parameters()
        if edge_current_mode not in ('static', 'dynamic', 'blend'):
            raise ValueError("edge_current_mode must be 'static', 'dynamic', or 'blend'")
        if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
            raise ValueError('blend must be finite and in [0, 1]')
        self.edge_current_mode = edge_current_mode
        self.blend = float(blend)
        super().__init__(params)
        self.force[0] = float(self.p.drive_amplitude)
        self.dynamic_link = np.copy(self.link_A)
        self._last_acc = self._acceleration()

    def _current_link(self):
        # Base initialization computes acceleration before dynamic_link is created.
        if not hasattr(self, 'dynamic_link'):
            return self.link_A
        if self.edge_current_mode == 'dynamic':
            return self.dynamic_link
        if self.edge_current_mode == 'blend':
            return (1.0 - self.blend) * self.link_A + self.blend * self.dynamic_link
        return self.link_A

    def step(self):
        dt = self.p.dt
        a = self._last_acc
        self.phi += dt * self.velocity + 0.5 * dt * dt * a
        self.velocity += 0.5 * dt * a
        i, j = self.edges.T
        edge_strain = np.zeros(self.n)
        np.add.at(edge_strain, i, self.phi[j] - self.phi[i])
        np.add.at(edge_strain, j, self.phi[i] - self.phi[j])
        target_density = self.p.strain_coupling * edge_strain + self.leak * np.cos(self.phi)
        self.density += dt * self.p.density_relaxation * (target_density - self.density)
        grad = np.zeros(self.n)
        np.add.at(grad, i, self.phi[j] - self.phi[i])
        np.add.at(grad, j, self.phi[j] - self.phi[i])
        target_p = np.zeros_like(self.polarization)
        target_p[:, 0] = grad + self.p.polarization_coupling * self.density
        self.polarization += dt * (target_p - self.polarization)
        if self.p.ring_dynamic_coupling > 0.0:
            tetra_mask = (self.edge_kind == 'tetra')
            mean_strain = float(np.mean(np.abs(edge_strain)))
            mean_pol = float(np.mean(np.linalg.norm(self.polarization, axis=1)))
            coupling_shift = self.p.ring_dynamic_coupling * 0.1 * (mean_strain + 0.5 * mean_pol)
            relax_rate = 0.5
            target_link = self.link_A + tetra_mask * coupling_shift
            self.dynamic_link += dt * relax_rate * (target_link - self.dynamic_link)
        anew = self._acceleration()
        self.velocity += 0.5 * dt * anew
        self._last_acc = anew
        self.time += dt
        if not all(np.all(np.isfinite(v)) for v in
                   (self.phi, self.velocity, self.density, self.polarization,
                    self.dynamic_link)):
            raise FloatingPointError('non-finite state; reduce dt or coupling strengths')
        return self.diagnostics()

    def diagnostics(self):
        result = super().diagnostics()
        static_flux = face_fluxes(self.link_A, self.edges, self._faces)
        dynamic_flux = face_fluxes(self.dynamic_link, self.edges, self._faces)
        result['ring_order_static'] = float(np.mean(np.cos(static_flux)))
        result['ring_order_dynamic'] = float(np.mean(np.cos(dynamic_flux)))
        result['ring_flux_rms'] = float(np.sqrt(np.mean(dynamic_flux ** 2)))
        result['dynamic_link_rms'] = float(np.sqrt(np.mean(self.dynamic_link ** 2)))
        result['edge_current_mode'] = self.edge_current_mode
        result['mean_energy_proxy_note'] = 'diagnostic_only_not_conserved'
        result['breakdown_flag'] = False
        return result


class RingFeedbackRegressionTests(unittest.TestCase):
    def test_zero_coupling_static_matches_baseline_trajectory(self):
        p = Parameters(n_cells=4, dt=0.01, seed=11, ring_dynamic_coupling=0.0)
        base = LocalSDFLattice(p)
        ring = DynamicRingSDFLattice(p, edge_current_mode='static')
        max_phi_error = 0.0
        max_velocity_error = 0.0
        for _ in range(30):
            base.step()
            ring.step()
            max_phi_error = max(max_phi_error,
                                float(np.max(np.abs(base.phi - ring.phi))))
            max_velocity_error = max(max_velocity_error,
                                     float(np.max(np.abs(base.velocity - ring.velocity))))
        self.assertTrue(np.array_equal(base.phi, ring.phi))
        self.assertTrue(np.array_equal(base.velocity, ring.velocity))
        self.assertTrue(np.array_equal(base.density, ring.density))
        self.assertEqual(max_phi_error, 0.0)
        self.assertEqual(max_velocity_error, 0.0)

    def test_dynamic_current_changes_phase_acceleration(self):
        p = Parameters(n_cells=4, dt=0.01, seed=11, ring_dynamic_coupling=1.2,
                       drive_amplitude=0.7)
        static = DynamicRingSDFLattice(p, edge_current_mode='static')
        dynamic = DynamicRingSDFLattice(p, edge_current_mode='dynamic')
        initial_difference = float(np.max(np.abs(static._last_acc - dynamic._last_acc)))
        for _ in range(120):
            static.step()
            dynamic.step()
        acceleration_difference = float(np.max(np.abs(static._last_acc - dynamic._last_acc)))
        phase_difference = float(np.max(np.abs(static.phi - dynamic.phi)))
        link_difference = float(np.max(np.abs(static.dynamic_link - static.link_A)))
        self.assertEqual(initial_difference, 0.0)
        self.assertGreater(acceleration_difference, 1e-8)
        self.assertGreater(phase_difference, 1e-8)
        self.assertGreater(link_difference, 0.0)

    def test_face_orientation_reversal_sign_and_invariant_metrics(self):
        p = Parameters(n_cells=3, seed=13, ring_dynamic_coupling=1.0)
        sim = DynamicRingSDFLattice(p, edge_current_mode='dynamic')
        sim.run(80)
        original = sim._faces.copy()
        reversed_faces = original[:, [0, 2, 1]]
        flux = face_fluxes(sim.dynamic_link, sim.edges, original)
        reversed_flux = face_fluxes(sim.dynamic_link, sim.edges, reversed_faces)
        self.assertTrue(np.allclose(reversed_flux, -flux, rtol=0.0, atol=1e-14))
        order_a = float(np.mean(np.cos(flux)))
        order_b = float(np.mean(np.cos(reversed_flux)))
        rms_a = float(np.sqrt(np.mean(flux ** 2)))
        rms_b = float(np.sqrt(np.mean(reversed_flux ** 2)))
        self.assertAlmostEqual(order_a, order_b, places=14)
        self.assertAlmostEqual(rms_a, rms_b, places=14)
        self.assertTrue(np.allclose(np.cos(flux), np.cos(reversed_flux), atol=1e-14))

    def test_blend_equation_and_validation(self):
        p = Parameters(n_cells=2, seed=5, ring_dynamic_coupling=0.8)
        sim = DynamicRingSDFLattice(p, edge_current_mode='blend', blend=0.25)
        sim.run(40)
        expected = 0.75 * sim.link_A + 0.25 * sim.dynamic_link
        self.assertTrue(np.array_equal(sim._current_link(), expected))
        with self.assertRaises(ValueError):
            DynamicRingSDFLattice(p, edge_current_mode='unknown')
        with self.assertRaises(ValueError):
            DynamicRingSDFLattice(p, edge_current_mode='blend', blend=1.1)


def run_harness():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RingFeedbackRegressionTests)
    result = unittest.TestResult()
    suite.run(result)
    case_results = []
    for test, err in result.failures + result.errors:
        case_results.append({'test': str(test), 'status': 'FAIL', 'detail': err})
    failed_ids = {str(test) for test, _ in result.failures + result.errors}
    for name in ('test_zero_coupling_static_matches_baseline_trajectory',
                 'test_dynamic_current_changes_phase_acceleration',
                 'test_face_orientation_reversal_sign_and_invariant_metrics',
                 'test_blend_equation_and_validation'):
        test_id = 'RingFeedbackRegressionTests.' + name
        if test_id not in failed_ids:
            case_results.append({'test': test_id, 'status': 'PASS'})

    # Recompute key numeric evidence for a machine-readable report.
    p0 = Parameters(n_cells=4, dt=0.01, seed=11, ring_dynamic_coupling=0.0)
    base = LocalSDFLattice(p0)
    ring = DynamicRingSDFLattice(p0, edge_current_mode='static')
    for _ in range(30):
        base.step()
        ring.step()
    zero_error = float(np.max(np.abs(base.phi - ring.phi)))
    p1 = Parameters(n_cells=4, dt=0.01, seed=11, ring_dynamic_coupling=1.2,
                    drive_amplitude=0.7)
    st = DynamicRingSDFLattice(p1, edge_current_mode='static')
    dy = DynamicRingSDFLattice(p1, edge_current_mode='dynamic')
    for _ in range(120):
        st.step()
        dy.step()
    accel_delta = float(np.max(np.abs(st._last_acc - dy._last_acc)))
    phase_delta = float(np.max(np.abs(st.phi - dy.phi)))
    p2 = Parameters(n_cells=3, seed=13, ring_dynamic_coupling=1.0)
    orient_sim = DynamicRingSDFLattice(p2, edge_current_mode='dynamic')
    orient_sim.run(80)
    flux_a = face_fluxes(orient_sim.dynamic_link, orient_sim.edges, orient_sim._faces)
    flux_b = face_fluxes(orient_sim.dynamic_link, orient_sim.edges,
                         orient_sim._faces[:, [0, 2, 1]])
    report = {
        'source_logic': 'SDF-Vortex-(3).zip baseline and dynamic-ring source',
        'all_tests_passed': result.wasSuccessful(),
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'test_results': case_results,
        'edge_current_architecture': {
            'default_mode': 'static',
            'static_equation': 'link_eff = link_A',
            'dynamic_equation': 'link_eff = dynamic_link',
            'blend_equation': 'link_eff = (1 - blend) * link_A + blend * dynamic_link',
            'ring_dynamic_coupling_role': 'controls dynamic_link relaxation, not blend fraction'
        },
        'regression_metrics': {
            'zero_coupling_static_max_phase_error_after_30_steps': zero_error,
            'nonzero_coupling_dynamic_vs_static_max_acceleration_difference': accel_delta,
            'nonzero_coupling_dynamic_vs_static_max_phase_difference': phase_delta,
            'face_reversal_max_flux_sum_abs': float(np.max(np.abs(flux_a + flux_b))),
            'face_reversal_cosine_order_difference': float(abs(np.mean(np.cos(flux_a)) -
                                                                  np.mean(np.cos(flux_b)))),
            'face_reversal_flux_rms_difference': float(abs(np.sqrt(np.mean(flux_a ** 2)) -
                                                                  np.sqrt(np.mean(flux_b ** 2))))
        },
        'energy_statement': 'mean_energy_proxy is diagnostic only; conservation is not claimed.'
    }
    out = Path('/mnt/data/foundation_followup/foundation_followup_results.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='ascii')
    print(json.dumps(report, indent=2, sort_keys=True))
    print('JSON_OUTPUT=' + str(out))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(run_harness())
