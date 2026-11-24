"""Demonstrate repeated node execution outside the DAG scheduler.

Run from the SDF_Lattice_Master_v0.3.6 project root, or pass --output.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from sdf_lattice.nodes.strain import StrainTensorNode
from sdf_lattice.nodes.permittivity import PermittivityEnvelopeNode
from sdf_lattice.nodes.closure_constraints import ClosureConstraintsNode


def run_loop(steps: int = 100, output: Path = Path("loop_telemetry.csv")) -> list[dict]:
    strain_node = StrainTensorNode("strain", config={"critical_threshold": 0.005})
    permittivity_node = PermittivityEnvelopeNode(
        "permittivity", params={"radiative_leak_rate": 0.85}
    )
    closure_node = ClosureConstraintsNode("closure")

    records = []
    for step in range(steps):
        # A small, smoothly varying symmetric tensor used as the step's external drive.
        phase = 2.0 * np.pi * step / max(steps, 1)
        amplitude = 0.012 + 0.006 * np.sin(phase)
        strain_out = strain_node.compute({
            "tensor": [[amplitude, 0.002, 0.0],
                       [0.002, -0.4 * amplitude, 0.0],
                       [0.0, 0.0, 0.2 * amplitude]],
        })

        # Pass the strain node's fields required by the permittivity node.
        permittivity_in = {
            key: strain_out[key]
            for key in ("trace", "frobenius_dev", "is_critical")
        }
        permittivity_out = permittivity_node.compute(permittivity_in)

        # Closure consumes the strain output and the previous node's effective
        # envelope; radiative_flux is carried through for per-step accounting.
        closure_in = {
            "strain_tensor": strain_out["strain_tensor"],
            "permittivity_envelope": permittivity_out["effective_permittivity"],
            "coupling_strength": 1.0,
            "radiative_flux": permittivity_out["radiative_flux"],
        }
        closure_out = closure_node.compute(closure_in)
        records.append({
            "step": step,
            "radiative_flux": permittivity_out["radiative_flux"],
            "closure_status": closure_out["closure_status"],
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "radiative_flux", "closure_status"])
        writer.writeheader()
        writer.writerows(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("loop_telemetry.csv"))
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    records = run_loop(args.steps, args.output)
    print(f"Ran {len(records)} steps; telemetry saved to {args.output}")
    print(f"First: {records[0]}")
    print(f"Last:  {records[-1]}")


if __name__ == "__main__":
    main()
