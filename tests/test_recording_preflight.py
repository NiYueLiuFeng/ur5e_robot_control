import unittest


class RecordingPreflightTest(unittest.TestCase):
    def test_check_imports_reports_missing_and_available_modules(self):
        from ur5e_control.data_collection.preflight import check_imports

        def fake_find_spec(name):
            return object() if name == "available_module" else None

        results = check_imports(
            ["available_module", "missing_module"],
            find_spec=fake_find_spec,
        )

        self.assertTrue(results[0].ok)
        self.assertEqual(results[0].name, "import available_module")
        self.assertFalse(results[1].ok)
        self.assertIn("not importable", results[1].detail)

    def test_collect_preflight_builds_lerobot_dependency_checks(self):
        from ur5e_control.data_collection.preflight import collect_recording_preflight

        seen_imports = []

        def fake_find_spec(name):
            seen_imports.append(name)
            return object()

        results = collect_recording_preflight(
            gello_port="/tmp/nonexistent-gello-port",
            robot_ip="127.0.0.1",
            agent_name="gello",
            check_robot_tcp=False,
            gripper_port="/tmp/nonexistent-pgc-port",
            realsense_ids={"333422304095": "wrist"},
            find_spec=fake_find_spec,
        )

        self.assertIn("numpy", seen_imports)
        self.assertIn("lerobot", seen_imports)
        self.assertIn("pyrealsense2", seen_imports)
        self.assertTrue(
            any(
                result.name == "gello port /tmp/nonexistent-gello-port"
                for result in results
            )
        )
        self.assertTrue(
            any(
                result.name == "pgc gripper port /tmp/nonexistent-pgc-port"
                for result in results
            )
        )
        self.assertTrue(
            any(
                result.name == "configured realsense cameras"
                and "333422304095=wrist" in result.detail
                for result in results
            )
        )

    def test_collect_preflight_adds_spacemouse_dependency_checks(self):
        from ur5e_control.data_collection.preflight import collect_recording_preflight

        seen_imports = []

        def fake_find_spec(name):
            seen_imports.append(name)
            return object()

        results = collect_recording_preflight(
            gello_port="/tmp/nonexistent-gello-port",
            robot_ip="127.0.0.1",
            agent_name="spacemouse",
            spacemouse_device_path="/tmp/nonexistent-spacemouse",
            find_spec=fake_find_spec,
        )

        self.assertIn("dm_control", seen_imports)
        self.assertIn("pyspacemouse", seen_imports)
        self.assertIn("quaternion", seen_imports)
        self.assertFalse(
            any(
                result.name == "gello port /tmp/nonexistent-gello-port"
                for result in results
            )
        )
        self.assertTrue(
            any(
                result.name == "spacemouse device /tmp/nonexistent-spacemouse"
                for result in results
            )
        )
        self.assertTrue(
            any(
                result.name.startswith("spacemouse mujoco model ")
                for result in results
            )
        )

    def test_build_lerobot_record_command_sets_expected_environment(self):
        from ur5e_control.data_collection.preflight import build_lerobot_record_command

        command = build_lerobot_record_command(
            repo_id="chopintelligence/ur5e-pen-pick-place",
            root="/data/ur5e-pen",
            fps=20,
            config_path="configs/hardware/ur5e_d455.yaml",
        )

        self.assertIn(
            "UR5E_LEROBOT_REPO_ID=chopintelligence/ur5e-pen-pick-place",
            command,
        )
        self.assertIn("UR5E_LEROBOT_ROOT=/data/ur5e-pen", command)
        self.assertIn("UR5E_HARDWARE_CONFIG=configs/hardware/ur5e_d455.yaml", command)
        self.assertIn("uv run record-teleop", command)

    def test_format_preflight_report_returns_nonzero_when_blocked(self):
        from ur5e_control.data_collection.preflight import (
            PreflightResult,
            format_preflight_report,
        )

        report = format_preflight_report(
            [
                PreflightResult(name="import numpy", ok=True, detail="ok"),
                PreflightResult(name="import lerobot", ok=False, detail="not importable"),
            ]
        )

        self.assertIn("[ok] import numpy", report)
        self.assertIn("[blocked] import lerobot", report)


if __name__ == "__main__":
    unittest.main()
