import importlib.util
import unittest


class FakeDynamixelDriverTest(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("numpy is not installed")
        if importlib.util.find_spec("dynamixel_sdk") is None:
            self.skipTest("dynamixel_sdk is not installed")
        from ur5e_control.data_collection.teleop.gello.dynamixel.driver import FakeDynamixelDriver

        self.driver_cls = FakeDynamixelDriver

    def test_set_joints_updates_joint_state_when_torque_is_enabled(self):
        driver = self.driver_cls(ids=[1, 2])

        driver.set_torque_mode(True)
        driver.set_joints([1.57, 1.57])

        self.assertEqual(driver.get_joints().tolist(), [1.57, 1.57])

    def test_set_joints_rejects_wrong_length(self):
        driver = self.driver_cls(ids=[1, 2])

        with self.assertRaises(ValueError):
            driver.set_joints([1.57])

    def test_set_joints_rejects_commands_when_torque_is_disabled(self):
        driver = self.driver_cls(ids=[1, 2])

        with self.assertRaises(RuntimeError):
            driver.set_joints([1.57, 1.57])

    def test_torque_enabled_reflects_current_mode(self):
        driver = self.driver_cls(ids=[1, 2])

        self.assertFalse(driver.torque_enabled())
        driver.set_torque_mode(True)

        self.assertTrue(driver.torque_enabled())

    def test_get_joints_defaults_to_zero_positions(self):
        driver = self.driver_cls(ids=[1, 2])

        self.assertEqual(driver.get_joints().tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
