"""Dynamic-ring extension of the local SDF lattice model.

Pure ASCII/English code and comments to avoid encoding conflicts.
Based strictly on SDF-Vortex-(2).zip local lattice baseline.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
import sys
import importlib.util
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.join(CURRENT_DIR, "sdf_local_lattice.py")

if not os.path.exists(BASELINE_PATH):
    raise FileNotFoundError(f"Baseline file not found at: {BASELINE_PATH}")

SPEC = importlib.util.spec_from_file_location("base_sdf_module", BASELINE_PATH)
_BASE = importlib.util.module_from_spec(SPEC)
sys.modules["base_sdf_module"] = _BASE
SPEC.loader.exec_module(_BASE)

BaseParameters = _BASE.Parameters
BaseLocalSDFLattice = _BASE.LocalSDFLattice


@dataclass(frozen=True)
class Parameters(BaseParameters):
    drive_amplitude: float = 0.35
    ring_dynamic_coupling: float = 0.0

    def validate(self):
        super().validate()
        for name in ("drive_amplitude", "ring_dynamic_coupling"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        return self


class DynamicRingSDFLattice(BaseLocalSDFLattice):
    """Local lattice supporting variable drive amplitude and dynamic link variables."""

    def __init__(self, params: Parameters | None = None):
        if params is None:
            params = Parameters()
        super().__init__(params)

        # Apply configurable drive amplitude to node 0
        self.force[0] = float(self.p.drive_amplitude)

        # Dynamic link field initialized identical to static link_A
        self.dynamic_link = np.copy(self.link_A)
        self._last_acc = self._acceleration()

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

        # Dynamic ring relaxation towards static geometry plus local coupling
        if self.p.ring_dynamic_coupling > 0.0:
            tetra_mask = (self.edge_kind == "tetra")
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

        if not all(np.all(np.isfinite(v)) for v in (self.phi, self.velocity, self.density, self.polarization, self.dynamic_link)):
            raise FloatingPointError("Non-finite values encountered during integration step.")

        return self.diagnostics()

    def diagnostics(self):
        base_diag = super().diagnostics()

        edge_index = {tuple(e): k for k, e in enumerate(self.edges)}

        static_flux = []
        dyn_flux = []

        for a, b, c in self._faces:
            tot_static = 0.0
            tot_dyn = 0.0
            for u, v in ((a, b), (b, c), (c, a)):
                if u < v:
                    idx = edge_index[(u, v)]
                    tot_static += self.link_A[idx]
                    tot_dyn += self.dynamic_link[idx]
                else:
                    idx = edge_index[(v, u)]
                    tot_static -= self.link_A[idx]
                    tot_dyn -= self.dynamic_link[idx]

            static_flux.append(np.angle(np.exp(1j * tot_static)))
            dyn_flux.append(np.angle(np.exp(1j * tot_dyn)))

        static_flux_arr = np.asarray(static_flux, dtype=float)
        dyn_flux_arr = np.asarray(dyn_flux, dtype=float)

        base_diag["ring_order_static"] = float(np.mean(np.cos(static_flux_arr)))
        base_diag["ring_order_dynamic"] = float(np.mean(np.cos(dyn_flux_arr)))
        base_diag["ring_flux_rms"] = float(np.sqrt(np.mean(dyn_flux_arr ** 2)))
        base_diag["dynamic_link_rms"] = float(np.sqrt(np.mean(self.dynamic_link ** 2)))
        base_diag["breakdown_flag"] = False

        return base_diag


LocalSDFLattice = DynamicRingSDFLattice
