from pathlib import Path
import sys
import numpy as np

# تنظیم مسیر ریشهٔ پروژه
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sdf_lattice.core import Graph
from sdf_lattice.nodes.twist_relay import TwistRelayNode


def execute_weight_sweep():
    print("--- شروع تحلیل حساسیت و جاروب وزن‌ها (Weight Sweep) ---")

    weights = np.linspace(0.2, 2.0, 10)
    sweep_results = []

    for weight in weights:
        graph = Graph()

        # ساخت و افزودن نود با ضریب جفت‌شدگی متغیر
        relay_node = TwistRelayNode(
            "relay_1",
            {
                "coupling_factor": float(weight),
                "phase_delay_coeff": 0.05,
            },
        )
        graph.add_node(relay_node)

        # ساختار ورودی اولیه استاندارد گراف
        inputs = {
            "relay_1": {
                "local_phase": 0.05,
                "angular_velocity": [0.0, 0.0, 1.2],
            }
        }

        try:
            results = graph.run(inputs)
            relay_output = results.get("relay_1", {})

            torque = relay_output.get("induced_torque_vector")
            macro_twist = relay_output.get("macro_twist_magnitude")

            torque_z = None
            if isinstance(torque, (list, tuple)) and len(torque) == 3:
                torque_z = float(torque[2])

            sweep_results.append(
                {
                    "weight": float(weight),
                    "macro_twist": (
                        float(macro_twist)
                        if macro_twist is not None
                        else None
                    ),
                    "torque_z": torque_z,
                }
            )

        except Exception as exc:
            print(f"خطا در وزن {weight:.2f}: {exc}")

    print()
    print(f"{'Weight (w)':<12} | {'Macro Twist':<15} | {'Torque Z':<12}")
    print("-" * 45)

    for result in sweep_results:
        macro = result["macro_twist"]
        torque_z = result["torque_z"]

        macro_text = f"{macro:.6f}" if macro is not None else "ناموجود"
        torque_text = f"{torque_z:.6f}" if torque_z is not None else "ناموجود"

        print(
            f"{result['weight']:<12.2f} | "
            f"{macro_text:<15} | "
            f"{torque_text:<12}"
        )

    return sweep_results


if __name__ == "__main__":
    execute_weight_sweep()
