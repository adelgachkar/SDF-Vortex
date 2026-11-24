"""
Unit tests for TwistRelayNode.
Verifies phase relay computation, finite checks, wobble generation, and induced torque.
"""

import unittest
from sdf_lattice.nodes.twist_relay import TwistRelayNode


class TestTwistRelayNode(unittest.TestCase):
    
    def setUp(self):
        self.node = TwistRelayNode(node_id="test_twist_1", params={"angle_deficit": 0.12838834})
        
    def test_basic_computation(self):
        inputs = {
            "local_phase": 0.5,
            "angular_velocity": [1.0, 0.0, 10.0],
            "ambient_field": [0.0, 0.0, 1.0]
        }
        res = self.node.compute(inputs)
        
        self.assertIn("cascaded_phase", res)
        self.assertIn("wobble_vector", res)
        self.assertIn("macro_twist_magnitude", res)
        self.assertIn("induced_torque_vector", res)
        self.assertTrue(res["phenomenological"])
        
        # بررسی اینکه فاز آبشاری درست جمع شده است
        expected_phase = 0.5 + 0.12838834 * 1.0
        self.assertAlmostEqual(res["cascaded_phase"], expected_phase)
        self.assertEqual(len(res["wobble_vector"]), 3)
        self.assertEqual(len(res["induced_torque_vector"]), 3)

    def test_invalid_inputs(self):
        # بررسی خطای فاز نامعتبر
        with self.assertRaises(ValueError):
            self.node.compute({
                "local_phase": float('nan'),
                "angular_velocity": [1.0, 0.0, 1.0]
            })
            
        # بررسی خطای ابعاد بردار سرعت زاویه‌ای
        with self.assertRaises(ValueError):
            self.node.compute({
                "local_phase": 0.1,
                "angular_velocity": [1.0, 2.0]  # نامعتبر (فقط دو عنصر)
            })

    def test_missing_ports(self):
        with self.assertRaises(KeyError):
            self.node.compute({
                "local_phase": 0.1
                # عدم ارسال angular_velocity
            })


if __name__ == "__main__":
    unittest.main()
