"""Exploratory local SDF lattice toy model; not validated physics.

Run: python sdf_local_lattice.py --demo [--steps N] [--seed N]
Test: python sdf_local_lattice.py --self-test
Requires: Python >=3.9 and NumPy.
"""
from __future__ import annotations
from dataclasses import dataclass
import argparse
import math
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

    def validate(self):
        if not isinstance(self.n_cells, int) or self.n_cells < 2:
            raise ValueError('n_cells must be an integer >= 2')
        positive = ('dt','inertia','stiffness','density_relaxation')
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('damping','link_phase','ring_flux','leak_strength',
                     'density_coupling','strain_coupling','polarization_coupling'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        return self

class LocalSDFLattice:
    """A chain of tetrahedral cells with phenomenological scalar fields.

    Every cell has four local vertices and all six tetrahedron edges. Adjacent
    cells are coupled by one axial edge. Link phases define gauge-invariant
    phase differences; the oriented sum around each tetrahedral face is flux.
    This is a discrete toy analogue, not a derivation from electromagnetism.
    """
    def __init__(self, params=Parameters()):
        self.p = params.validate()
        self.n = 4 * params.n_cells
        rng = np.random.default_rng(params.seed)
        self.edges = []
        self.edge_kind = []
        for c in range(params.n_cells):
            base = 4*c
            for a in range(4):
                for b in range(a+1,4):
                    self.edges.append((base+a,base+b)); self.edge_kind.append('tetra')
        for c in range(params.n_cells-1):
            self.edges.append((4*c,4*(c+1))); self.edge_kind.append('axial')
        self.edges = np.asarray(self.edges, dtype=int)
        self.edge_kind = np.asarray(self.edge_kind)
        self.phi = rng.normal(0, 0.025, self.n)
        self.velocity = np.zeros(self.n)
        self.density = np.zeros(self.n)
        self.polarization = np.zeros((self.n,3))
        # Axial asymmetry is a bounded spatial source, not a conservation law.
        x = np.repeat(np.arange(params.n_cells), 4).astype(float)
        self.leak = params.leak_strength * np.exp(-x/max(1,params.n_cells-1))
        self.link_A = np.zeros(len(self.edges))
        self.link_A[self.edge_kind == 'tetra'] = params.ring_flux / 3.0
        # Axial drive initially excites one local node and launches a chain wave.
        self.force = np.zeros(self.n)
        self.force[0] = 0.35
        self.time = 0.0
        self._faces = []
        for c in range(params.n_cells):
            b=4*c
            self._faces.extend([(b,b+1,b+2),(b,b+1,b+3),(b,b+2,b+3),(b+1,b+2,b+3)])
        self._faces=np.asarray(self._faces,dtype=int)
        self._last_acc = self._acceleration()

    def _edge_current(self):
        i,j=self.edges.T
        d=self.phi[j]-self.phi[i]-self.link_A
        return self.p.stiffness*np.sin(d), d

    def _acceleration(self):
        cur,_=self._edge_current()
        net=np.zeros(self.n)
        i,j=self.edges.T
        np.add.at(net,i,cur); np.add.at(net,j,-cur)
        # Density-mediated local restoring response and fixed axial excitation.
        net -= self.p.density_coupling*self.density
        net += self.force
        return (net-self.p.damping*self.velocity)/self.p.inertia

    def step(self):
        dt=self.p.dt
        a=self._last_acc
        self.phi += dt*self.velocity + 0.5*dt*dt*a
        self.velocity += 0.5*dt*a
        # Relaxing density follows local strain and asymmetric leakage.
        i,j=self.edges.T
        edge_strain=np.zeros(self.n)
        np.add.at(edge_strain,i,self.phi[j]-self.phi[i])
        np.add.at(edge_strain,j,self.phi[i]-self.phi[j])
        target=self.p.strain_coupling*edge_strain + self.leak*np.cos(self.phi)
        self.density += dt*self.p.density_relaxation*(target-self.density)
        # Polarization aligns with local axial phase gradient and density gradient.
        grad=np.zeros(self.n)
        np.add.at(grad,i,self.phi[j]-self.phi[i]); np.add.at(grad,j,self.phi[j]-self.phi[i])
        targetP=np.zeros_like(self.polarization)
        targetP[:,0]=grad + self.p.polarization_coupling*self.density
        self.polarization += dt*(targetP-self.polarization)
        anew=self._acceleration()
        self.velocity += 0.5*dt*anew
        self._last_acc=anew
        self.time += dt
        if not all(np.all(np.isfinite(v)) for v in (self.phi,self.velocity,self.density,self.polarization)):
            raise FloatingPointError('non-finite state; reduce dt or coupling strengths')
        return self.diagnostics()

    def diagnostics(self):
        _,d=self._edge_current()
        # Oriented Wilson-loop-like face flux: sum link phase modulo 2pi.
        edge_index={tuple(e):k for k,e in enumerate(self.edges)}
        flux=[]
        for a,b,c in self._faces:
            total=0.
            for u,v in ((a,b),(b,c),(c,a)):
                if u<v: total += self.link_A[edge_index[(u,v)]]
                else: total -= self.link_A[edge_index[(v,u)]]
            flux.append(np.angle(np.exp(1j*total)))
        strain=float(np.sqrt(np.mean(d*d)))
        rho=float(np.mean(self.density))
        pol=float(np.mean(np.linalg.norm(self.polarization,axis=1)))
        ring_order=float(np.mean(np.cos(np.asarray(flux))))
        energy=float(0.5*self.p.inertia*np.mean(self.velocity**2) +
                     self.p.stiffness*np.mean(1-np.cos(d)))
        return {'time':self.time,'mean_density':rho,'rms_strain':strain,
                'mean_polarization':pol,'ring_order':ring_order,
                'mean_energy_proxy':energy,'max_abs_phase':float(np.max(np.abs(self.phi)))}

    def stability_report(self):
        # Conservative linearized graph bound, only a numerical timestep heuristic.
        deg=np.bincount(self.edges.ravel(), minlength=self.n)
        omega_bound=math.sqrt(2*self.p.stiffness*float(deg.max())/self.p.inertia)
        ratio=self.p.dt*omega_bound
        return {'dt_omega_bound':ratio,'heuristic_stable':ratio < 0.5,
                'note':'Numerical integration heuristic only; not a physical sampling law.'}

    def run(self, steps):
        if not isinstance(steps,int) or steps < 1: raise ValueError('steps must be >= 1')
        hist=[]
        for _ in range(steps): hist.append(self.step())
        return hist

def self_test():
    p=Parameters(n_cells=4,dt=0.01,seed=11)
    a=LocalSDFLattice(p); b=LocalSDFLattice(p)
    ha=a.run(30); hb=b.run(30)
    assert np.allclose(a.phi,b.phi) and np.isfinite(ha[-1]['mean_energy_proxy'])
    assert len(a.edges)==6*p.n_cells+p.n_cells-1
    assert abs(ha[-1]['ring_order']) <= 1.0+1e-12
    try: Parameters(dt=0).validate()
    except ValueError: pass
    else: raise AssertionError('invalid dt accepted')
    print('SELF-TEST PASS: reproducibility, topology, finite diagnostics, validation')

def demo(steps=400,seed=7):
    p=Parameters(seed=seed)
    sim=LocalSDFLattice(p)
    hist=sim.run(steps)
    print('Exploratory phenomenological model; not validated physics.')
    print('Numerical stability:',sim.stability_report())
    print('Initial:',{k:round(v,6) for k,v in hist[0].items()})
    print('Final:',{k:round(v,6) for k,v in hist[-1].items()})
    return sim,hist

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--demo',action='store_true'); ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--steps',type=int,default=400); ap.add_argument('--seed',type=int,default=7)
    args=ap.parse_args()
    if args.self_test: self_test()
    else: demo(args.steps,args.seed)
