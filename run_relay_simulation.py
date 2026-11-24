import numpy as np
from sdf_lattice.core import Graph
from sdf_lattice.registry import NodeRegistry, register_defaults

# ۱. راه‌اندازی و ثبت گره‌ها
registry = NodeRegistry()
register_defaults(registry)

# ۲. ایجاد گراف
graph = Graph()

# ۳. افزودن گره‌های گراف
# الف: یک گره فرضی برای تأمین ورودی‌ها (می‌توانست یک گره پیچیده‌تر باشد)
# ب: گره رله پیچشی (Twist Relay)
graph.add_node(registry.create("twist_relay", "planetary_relay_01", {
    "coupling_factor": 1.5,
    "phase_delay_coeff": 0.08
}))

# ۴. اجرای یک چرخه محاسباتی (شبیه‌سازی یک وضعیت لحظه‌ای)
# ورودی‌ها: فاز موضعی شبکه، سرعت زاویه‌ای کنونی، میدان محیطی
inputs = {
    "local_phase": 0.15, 
    "angular_velocity": [0.0, 0.0, 1.0], 
    "ambient_field": [0.0, 0.1, 0.9]
}

# اجرای محاسبه
results = graph.nodes["planetary_relay_01"].compute(inputs)

# ۵. نمایش خروجی‌های پدیدارشناختی
print("--- خروجی‌های شبیه‌سازی رله پیچشی ---")
print(f"فاز آبشاری (Cascaded Phase): {results['cascaded_phase']:.6f}")
print(f"میدان پیچش کلان (Macro Twist Magnitude): {results['macro_twist_magnitude']:.6f}")
print(f"گشتاور تحمیلی بر سیاره (Induced Torque Vector):")
print(f"  X: {results['induced_torque_vector'][0]:.6f}")
print(f"  Y: {results['induced_torque_vector'][1]:.6f}")
print(f"  Z: {results['induced_torque_vector'][2]:.6f}")
print("-------------------------------------")
