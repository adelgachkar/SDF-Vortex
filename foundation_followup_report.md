# Foundation Energy Follow-up

All results use the authoritative source modules in `/mnt/data/.sdf_vortex_validation_source`. The energy analysis is diagnostic-proxy behavior, not a physical energy conservation test.

## A. Energy-proxy trajectory

Configuration: n_cells=12, seed=7, dt=0.02, steps=400, drive_amplitude=0.35, ring_dynamic_coupling=0.35; remaining defaults.

```
{
  "absolute_drift": 0.009087814816627845,
  "final": 0.011361622432979949,
  "finite": true,
  "finite_count": 400,
  "first_five": [
    0.002273807616352104,
    0.002280789974793569,
    0.0022923176992771794,
    0.0023082675392136934,
    0.002328489339633397
  ],
  "initial": 0.002273807616352104,
  "last_five": [
    0.01120824087444381,
    0.01124441747718934,
    0.011282072167375676,
    0.011321158898037153,
    0.011361622432979949
  ],
  "linear_least_squares_slope": 0.0011565416220270608,
  "max": 0.011361622432979949,
  "max_absolute_one_step_increment": 6.166522706282235e-05,
  "min": 0.002273807616352104,
  "monotonic_decreasing": false,
  "monotonic_increasing": false,
  "relative_drift": 3.9967386648161254,
  "total_count": 400
}
```

## B. Ring-order ensemble

| Seed | Final ring_order_dynamic | Finite |
|---:|---:|:---:|
| 7 | 0.99717103760251857 | True |
| 42 | 0.99718452673981217 | True |
| 100 | 0.99717188695261239 | True |
| 200 | 0.99716947505662346 | True |
| 314 | 0.99717793087160189 | True |

Ensemble summary (sample standard deviation uses ddof=1):

```
{
  "coefficient_of_variation": 6.2478461654533305e-06,
  "max": 0.9971845267398122,
  "mean": 0.9971749714446337,
  "min": 0.9971694750566235,
  "sample_standard_deviation_ddof_1": 6.230195821626389e-06
}
```

## C. Relaxation and polarization sensitivity

These parameter variations are sensitivity/behavior tests and do not by themselves prove a thermodynamic dissipation law.

| Case | Status | Final energy | Drift | Delta final energy | Delta drift | Finite |
|---|---|---:|---:|---:|---:|:---:|
| baseline | ok | 0.011361622432979949 | 0.0090878148166278448 | 0 | 0 | True |
| density_relaxation_zero | error: ValueError: density_relaxation must be finite and positive | | | | | |
| density_relaxation_high | ok | 0.012547470147429841 | 0.01027366439687395 | 0.0011858477144498919 | 0.0011858495802461056 | True |
| polarization_coupling_zero | ok | 0.011361622432979949 | 0.0090878148166278448 | 0 | 0 | True |
| polarization_coupling_high | ok | 0.011361622432979949 | 0.0090878148166278448 | 0 | 0 | True |

Full case metrics are in the comparison CSV and JSON.

## D. Source checks

```
{
  "DynamicRingSDFLattice_overrides__acceleration": false,
  "dynamic_link_passed_into_acceleration_path": false,
  "mean_energy_proxy_expression": "float(0.5 * self.p.inertia * np.mean(self.velocity ** 2) + self.p.stiffness * np.mean(1 - np.cos(d)))",
  "mean_energy_proxy_uses_dynamic_link": false,
  "mean_energy_proxy_uses_link_A": false,
  "note": "Dynamic extension inherits acceleration; no dynamic_link dependency is present in the base acceleration implementation."
}
```

## Self-test

Command: `/usr/local/bin/python /mnt/data/.sdf_vortex_validation_source/sdf_local_lattice.py --self-test`
Working directory: `/mnt/data/.sdf_vortex_validation_source`
Exit code: 0

Stdout:

```
SELF-TEST PASS: reproducibility, topology, finite diagnostics, validation
```

Stderr:

```

```

## Limitations

- The proxy is not calibrated physical energy.
- Drive, damping, relaxation, and feedback can change it.
- Parameter variations are sensitivity/behavior tests and do not by themselves prove a thermodynamic dissipation law.
- Dynamic_link is not in the acceleration path if confirmed by the structural checks above.
