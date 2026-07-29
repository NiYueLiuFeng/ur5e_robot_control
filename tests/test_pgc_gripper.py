import unittest
import importlib.util
from pathlib import Path


class PGCGripperCommandRateTest(unittest.TestCase):
    def test_repeated_target_within_tolerance_is_not_resent(self):
        pgc_module = load_pgc_module()

        fake_gripper = FakeDHGripper()
        clock = FakeClock()

        with patched_gripper_module(pgc_module, fake_gripper):
            gripper = pgc_module.PGCGripper(
                port="/dev/fake-pgc",
                position_tolerance=10,
                min_command_interval_s=0.05,
                position_read_interval_s=0.1,
                clock=clock,
            )
            gripper.grip(500)
            gripper.grip(505)
            clock.advance(0.1)
            gripper.grip(506)
            gripper.grip(520)

        self.assertEqual(fake_gripper.target_positions, [500, 520])

    def test_changed_target_waits_for_minimum_command_interval(self):
        pgc_module = load_pgc_module()

        fake_gripper = FakeDHGripper()
        clock = FakeClock()

        with patched_gripper_module(pgc_module, fake_gripper):
            gripper = pgc_module.PGCGripper(
                port="/dev/fake-pgc",
                position_tolerance=10,
                min_command_interval_s=0.05,
                position_read_interval_s=0.1,
                clock=clock,
            )
            gripper.grip(500)
            gripper.grip(700)
            clock.advance(0.05)
            gripper.grip(700)

        self.assertEqual(fake_gripper.target_positions, [500, 700])

    def test_position_reads_are_cached_between_poll_intervals(self):
        pgc_module = load_pgc_module()

        fake_gripper = FakeDHGripper(current_positions=[111, 222])
        clock = FakeClock()

        with patched_gripper_module(pgc_module, fake_gripper):
            gripper = pgc_module.PGCGripper(
                port="/dev/fake-pgc",
                position_tolerance=10,
                min_command_interval_s=0.05,
                position_read_interval_s=0.1,
                clock=clock,
            )
            first = gripper.get_gripper_position()
            second = gripper.get_gripper_position()
            clock.advance(0.11)
            third = gripper.get_gripper_position()

        self.assertEqual((first, second, third), (111, 111, 222))
        self.assertEqual(fake_gripper.position_read_count, 2)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeDHGripper:
    def __init__(self, current_positions=None):
        self.current_positions = list(current_positions or [0])
        self.target_positions = []
        self.target_forces = []
        self.target_speeds = []
        self.position_read_count = 0
        self.open_calls = []
        self.initialized = False

    def open(self, port, baudrate):
        self.open_calls.append((port, baudrate))

    def Initialization(self):
        self.initialized = True

    def GetInitState(self):
        return 1 if self.initialized else 0

    def SetTargetForce(self, force):
        self.target_forces.append(force)

    def SetTargetSpeed(self, speed):
        self.target_speeds.append(speed)

    def SetTargetPosition(self, position):
        self.target_positions.append(position)

    def GetCurrentPosition(self):
        self.position_read_count += 1
        if len(self.current_positions) > 1:
            return self.current_positions.pop(0)
        return self.current_positions[0]


class patched_gripper_module:
    def __init__(self, pgc_module, fake_gripper):
        self.pgc_module = pgc_module
        self.fake_gripper = fake_gripper
        self.original_loader = pgc_module._load_dh_gripper_module

    def __enter__(self):
        fake_module = FakeDHModule(self.fake_gripper)
        self.pgc_module._load_dh_gripper_module = lambda sdk_path=None: fake_module
        return fake_module

    def __exit__(self, exc_type, exc, tb):
        self.pgc_module._load_dh_gripper_module = self.original_loader


class FakeDHModule:
    def __init__(self, fake_gripper):
        self.fake_gripper = fake_gripper

    def dh_modbus_gripper(self):
        return self.fake_gripper


def load_pgc_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "ur5e_control"
        / "hardware"
        / "grippers"
        / "pgc.py"
    )
    spec = importlib.util.spec_from_file_location("test_pgc_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
