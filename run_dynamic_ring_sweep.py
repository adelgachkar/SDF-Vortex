"""Run dynamic-ring parameter sweeps and generate CSV, PNG, and PDF outputs.

Pure ASCII/English script without localized strings or custom font requirements.
"""
from __future__ import annotations
from pathlib import Path
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from sdf_local_lattice_dynamic_ring import Parameters, LocalSDFLattice

CSV_OUT = HERE / "adel_dynamic_ring_sweep.csv"
PNG_OUT = HERE / "adel_dynamic_ring_sweep.png"
PDF_OUT = HERE / "adel_dynamic_ring_research_note.pdf"

DRIVE_VALUES = [0.05, 0.15, 0.35, 0.70, 1.20, 2.00]
COUPLING_VALUES = [0.0, 0.1, 0.35, 0.7, 1.2]
STEPS = 400


def run_single(label: str, p: Parameters) -> dict:
    sim = LocalSDFLattice(p)
    try:
        history = sim.run(STEPS)
        last = history[-1]
        status = "ok"
    except (FloatingPointError, OverflowError, ValueError) as exc:
        last = sim.diagnostics()
        last["breakdown_flag"] = True
        last["error"] = f"{type(exc).__name__}: {exc}"
        status = "breakdown"

    row = {
        "sweep": label,
        "drive_amplitude": p.drive_amplitude,
        "ring_dynamic_coupling": p.ring_dynamic_coupling,
        "seed": p.seed,
        "steps": STEPS,
        "status": status,
    }
    row.update(last)
    return row


def make_plot(rows: list[dict]):
    d = [r for r in rows if r["sweep"] == "drive"]
    c = [r for r in rows if r["sweep"] == "coupling"]

    x_d = [r["drive_amplitude"] for r in d]
    x_c = [r["ring_dynamic_coupling"] for r in c]

    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    ax[0, 0].plot(x_d, [r["ring_order_static"] for r in d], "o-", label="static")
    ax[0, 0].plot(x_d, [r["ring_order_dynamic"] for r in d], "s-", label="dynamic")
    ax[0, 0].set(xlabel="Drive amplitude", ylabel="Ring order", title="Dynamic vs Static Ring Order")
    ax[0, 0].grid(True, linestyle="--", alpha=0.5)
    ax[0, 0].legend()

    ax[0, 1].plot(x_d, [r["mean_polarization"] for r in d], "o-", color="tab:blue", label="polarization")
    ax[0, 1].set(xlabel="Drive amplitude", ylabel="Mean polarization", title="Polarization and Energy vs Drive")
    ax[0, 1].grid(True, linestyle="--", alpha=0.5)
    a2 = ax[0, 1].twinx()
    a2.plot(x_d, [r["mean_energy_proxy"] for r in d], "s--", color="tab:red", label="energy proxy")
    a2.set_ylabel("Energy Proxy")

    ax[1, 0].plot(x_c, [r["ring_order_static"] for r in c], "o-", label="static")
    ax[1, 0].plot(x_c, [r["ring_order_dynamic"] for r in c], "s-", label="dynamic")
    ax[1, 0].set(xlabel="Dynamic coupling", ylabel="Ring order", title="Coupling Sweep Response")
    ax[1, 0].grid(True, linestyle="--", alpha=0.5)
    ax[1, 0].legend()

    ax[1, 1].plot(x_d, [r["ring_flux_rms"] for r in d], "o-", label="flux RMS")
    ax[1, 1].plot(x_d, [r["dynamic_link_rms"] for r in d], "s-", label="dynamic link RMS")
    ax[1, 1].plot(x_d, [float(r["breakdown_flag"]) for r in d], "x--", label="breakdown flag")
    ax[1, 1].set(xlabel="Drive amplitude", ylabel="Indicator", title="Breakdown and Metric Indicators")
    ax[1, 1].grid(True, linestyle="--", alpha=0.5)
    ax[1, 1].legend()

    fig.suptitle("ADEL Dynamic-Ring Sweep (Simulation Summary)")
    fig.savefig(PNG_OUT, dpi=160)
    plt.close(fig)


def make_pdf(rows: list[dict]):
    d = [r for r in rows if r["sweep"] == "drive"]
    c = [r for r in rows if r["sweep"] == "coupling"]

    pages = [
        (
            "ADEL Dynamic Ring Simulation Report",
            [
                "Phenomenological local lattice exploration summary.",
                "Baseline reference: SDF-Vortex-(2).zip local lattice model.",
                "Parameters sweep: Drive amplitude and dynamic coupling variation.",
                "Results reflect numerical behavior of the model under specified constraints.",
            ],
        ),
        (
            "Sweep Protocol and Observable Definitions",
            [
                "Static ring order: Mean cosine of face flux derived from invariant link_A.",
                "Dynamic ring order: Computed from time-evolving dynamic link variables.",
                "Drive sweep range: [0.05, 0.15, 0.35, 0.70, 1.20, 2.00] with seed=7.",
                "Coupling sweep range: [0.00, 0.10, 0.35, 0.70, 1.20] at drive=0.35.",
                "Duration: 400 discrete simulation steps per run.",
            ],
        ),
        (
            "Key Numerical Outputs",
            [
                "Detailed execution results per test case:",
            ]
            + [
                f"Drive={r['drive_amplitude']:.2f} -> static={r['ring_order_static']:.4f}, "
                f"dynamic={r['ring_order_dynamic']:.4f}, pol={r['mean_polarization']:.4f}, "
                f"energy={r['mean_energy_proxy']:.4e}, breakdown={r['breakdown_flag']}"
                for r in d
            ]
            + [
                f"Coupling={r['ring_dynamic_coupling']:.2f} -> dynamic={r['ring_order_dynamic']:.4f}, "
                f"dynamic_rms={r['dynamic_link_rms']:.4f}, breakdown={r['breakdown_flag']}"
                for r in c
            ],
        ),
    ]

    with PdfPages(PDF_OUT) as pdf:
        for title, lines in pages:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.1, 0.92, title, fontsize=16, weight="bold")
            y = 0.85
            for line in lines:
                fig.text(0.1, y, line, fontsize=10, family="monospace")
                y -= 0.045
            pdf.savefig(fig)
            plt.close(fig)


def run():
    rows = [run_single("drive", Parameters(drive_amplitude=v, seed=7)) for v in DRIVE_VALUES]
    rows += [
        run_single("coupling", Parameters(drive_amplitude=0.35, ring_dynamic_coupling=v, seed=7))
        for v in COUPLING_VALUES
    ]

    fields = [
        "sweep",
        "drive_amplitude",
        "ring_dynamic_coupling",
        "seed",
        "steps",
        "status",
        "time",
        "mean_density",
        "rms_strain",
        "mean_polarization",
        "ring_order",
        "ring_order_static",
        "ring_order_dynamic",
        "ring_flux_rms",
        "dynamic_link_rms",
        "breakdown_flag",
        "mean_energy_proxy",
        "max_abs_phase",
        "error",
    ]

    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])

    make_plot(rows)
    make_pdf(rows)

    print("WROTE", CSV_OUT)
    print("WROTE", PNG_OUT)
    print("WROTE", PDF_OUT)

    for r in rows:
        print(
            f"{r['sweep']:<8} drive={r['drive_amplitude']:<5} coupling={r['ring_dynamic_coupling']:<5} "
            f"static={r['ring_order_static']:.6f} dynamic={r['ring_order_dynamic']:.6f} "
            f"pol={r['mean_polarization']:.6f} link_rms={r['dynamic_link_rms']:.6f} "
            f"breakdown={r['breakdown_flag']}"
        )
    return rows


if __name__ == "__main__":
    run()
