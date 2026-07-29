import unittest


class TeleopAgentFactoryTest(unittest.TestCase):
    def test_factory_creates_gello_agent_from_agent_config(self):
        from ur5e_control.data_collection.teleop.agents.factory import (
            create_teleop_agent,
        )
        from ur5e_control.runtime.config import AgentRuntimeConfig

        calls = []

        def fake_gello_factory(**kwargs):
            calls.append(kwargs)
            return "gello-agent"

        agent = create_teleop_agent(
            AgentRuntimeConfig(name="gello", gello_port="/dev/serial/by-id/gello"),
            gello_factory=fake_gello_factory,
        )

        self.assertEqual(agent, "gello-agent")
        self.assertEqual(calls, [{"port": "/dev/serial/by-id/gello"}])

    def test_factory_creates_spacemouse_agent_from_agent_config(self):
        from ur5e_control.data_collection.teleop.agents.factory import (
            create_teleop_agent,
        )
        from ur5e_control.runtime.config import (
            AgentRuntimeConfig,
            SpaceMouseRuntimeConfig,
        )

        calls = []

        def fake_spacemouse_factory(**kwargs):
            calls.append(kwargs)
            return "spacemouse-agent"

        agent = create_teleop_agent(
            AgentRuntimeConfig(
                name="spacemouse",
                robot="ur5",
                spacemouse=SpaceMouseRuntimeConfig(
                    device_path="/dev/hidraw0",
                    translation_scale=0.08,
                    angle_scale=0.3,
                    rotation_mode="rpy",
                    invert_button=True,
                    gripper_open=1000,
                    gripper_closed=0,
                ),
            ),
            spacemouse_factory=fake_spacemouse_factory,
        )

        self.assertEqual(agent, "spacemouse-agent")
        self.assertEqual(
            calls,
            [
                {
                    "robot_type": "ur5",
                    "device_path": "/dev/hidraw0",
                    "translation_scale": 0.08,
                    "angle_scale": 0.3,
                    "invert_control": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                    "rotation_mode": "rpy",
                    "invert_button": True,
                    "gripper_open": 1000,
                    "gripper_closed": 0,
                    "verbose": False,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
