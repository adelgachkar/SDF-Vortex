"""
Twist Relay Node: Cascaded Angular Deficit & Planetary Torque Imposition
Phenomenological operator modeling the hand-to-hand phase relay across a semi-crystalline 
lattice, calculating the macro-scale twist field and its induced torque/wobble on rotating bodies.
"""

from typing import Dict, Any, Sequence
import math

from .base import Node


class TwistRelayNode(Node):
    """
    مبدلِ رلهٔ پیچشی (Twist Relay) و انتقالِ کسریِ زاویه‌ای به مقیاس کلان و تلوتلوخوردنِ فرفره/سیاره.
    """
    
    type_name = "twist_relay"
    
    def __init__(self, node_id: str, params: Dict[str, Any] = None):
        super().__init__(node_id, params)
        # پیش‌فرض کسری زاویه برای 5 تترا: 2*pi - 5*arccos(1/3) ~ 7.356103 degrees ~ 0.128388 rad
        self.default_angle_deficit = 0.12838834  # radians
        
        # استخراج پارامترها با مقادیر پیش‌فرض امن
        self.angle_deficit = float(self.params.get("angle_deficit", self.default_angle_deficit))
        self.coupling_factor = float(self.params.get("coupling_factor", 1.0))
        self.phase_delay_coeff = float(self.params.get("phase_delay_coeff", 0.05))
        
    def _validate_vector(self, name: str, val: Any) -> Sequence[float]:
        if not isinstance(val, (list, tuple)) or len(val) != 3:
            raise ValueError(f"Port '{name}' must be a sequence of 3 floats.")
        resolved = []
        for x in val:
            if not isinstance(x, (int, float)) or not math.isfinite(x):
                raise ValueError(f"Port '{name}' elements must be finite numbers.")
            resolved.append(float(x))
        return resolved

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # اعتبارسنجی ورودی‌های اجباری
        if "local_phase" not in inputs or "angular_velocity" not in inputs:
            raise KeyError("TwistRelayNode requires 'local_phase' and 'angular_velocity' inputs.")
            
        local_phase = float(inputs["local_phase"])
        if not math.isfinite(local_phase):
            raise ValueError("Input 'local_phase' must be a finite number.")
            
        omega_in = self._validate_vector("angular_velocity", inputs["angular_velocity"])
        
        # اختیاری: بردار میدان پایه یا نویز محیطی
        ambient_field = inputs.get("ambient_field", [0.0, 0.0, 1.0])
        ambient_vec = self._validate_vector("ambient_field", ambient_field)
        
        # 1. محاسبه رله فاز آبشاری و انباشت کسری زاویه
        cascaded_phase = local_phase + self.angle_deficit * self.coupling_factor
        
        # 2. احتساب عقب‌ماندگی فاز (Retarded phase lag) و اثر تلوتلوخوردن (Wobble)
        wobble_x = omega_in[0] - self.phase_delay_coeff * cascaded_phase * ambient_vec[1]
        wobble_y = omega_in[1] + self.phase_delay_coeff * cascaded_phase * ambient_vec[0]
        wobble_z = omega_in[2] * math.cos(self.phase_delay_coeff * cascaded_phase)
        
        wobble_vector = [wobble_x, wobble_y, wobble_z]
        
        # 3. محاسبه میدان پیچش کلان (Macro Twist Field)
        macro_twist_magnitude = abs(cascaded_phase) * math.sqrt(sum(w**2 for w in wobble_vector))
        macro_twist_vector = [
            w * abs(cascaded_phase) * self.coupling_factor for w in wobble_vector
        ]
        
        # 4. گشتاور تحمیلی بر جسم دوار (Induced Planetary/Top Torque)
        induced_torque_vector = [
            macro_twist_vector[0] * 0.5,
            macro_twist_vector[1] * 0.5,
            macro_twist_vector[2] * 0.8
        ]
        
        return {
            "cascaded_phase": cascaded_phase,
            "wobble_vector": wobble_vector,
            "macro_twist_magnitude": macro_twist_magnitude,
            "macro_twist_vector": macro_twist_vector,
            "induced_torque_vector": induced_torque_vector,
            "phenomenological": True
        }
