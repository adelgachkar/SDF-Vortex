# Foundation Follow-up

This report uses the local dynamic-ring extension and a 400-step run. The model is an exploratory phenomenological toy model, not validated physics.

## Coupling and Acceleration

`ring_dynamic_coupling=0` leaves the dynamic link at its static initialization; positive coupling evolves it. The acceleration implementation inherits the base model and uses `link_A`, not `dynamic_link`. Directly perturbing only `dynamic_link` therefore does not affect acceleration.

- Link RMS at zero coupling: 0.062092042056506633
- Link RMS at coupling 0.35: 0.070074212205365305
- Dynamic link affects acceleration: False

## Multi-Seed Ensemble

Sample standard deviation uses ddof=1; CV is sample standard deviation divided by absolute mean.

| Seed | Final ring order | Final energy proxy | Dynamic-link RMS |
|---:|---:|---:|---:|
| 7 | 0.99717103760251857 | 0.011361622432979949 | 0.070074212205365305 |
| 42 | 0.99718452673981217 | 0.011225660183366615 | 0.069906869009718237 |
| 100 | 0.99717188695261239 | 0.011576069980097692 | 0.070063687128436761 |

Ring order: mean=0.99717581709831438, std=7.5547164182746647e-06, CV=7.576112746354161e-06.
Energy proxy: mean=0.011387784198814753, std=0.00017666376228580346, CV=0.015513444863504765.

## Orientation and Gauge Checks

- Global phase shift ring-order delta: 0
- Global phase shift RMS-strain delta: 0
- Global phase shift energy-proxy delta: 1.73e-18
- Face-orientation reversal cosine-order delta: 0
- Face-orientation reversal flux-RMS delta: 0

All invariance checks passed at absolute tolerance 1e-12. A global phase shift preserves edge differences; reversing face orientation flips flux signs while preserving cosine order and RMS.

## Outputs

- JSON: `/mnt/data/foundation_followup/foundation_followup_results.json`
- CSV: `/mnt/data/foundation_followup/ring_order_ensemble.csv`
- Report: `/mnt/data/foundation_followup/foundation_followup_report.md`
