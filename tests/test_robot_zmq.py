import unittest


class ZMQClientRobotTest(unittest.TestCase):
    def test_request_resets_socket_after_timeout(self):
        import zmq

        from ur5e_control.runtime.transport.robot_zmq import ZMQClientRobot

        client = ZMQClientRobot.__new__(ZMQClientRobot)
        client._socket = TimeoutSocket(zmq.Again())
        client.reset_count = 0

        def reset_socket():
            client.reset_count += 1

        client._reset_socket = reset_socket

        with self.assertRaisesRegex(RuntimeError, "timeout"):
            client.get_joint_state()

        self.assertEqual(client.reset_count, 1)


class TimeoutSocket:
    def __init__(self, exc):
        self.exc = exc

    def send(self, _message):
        pass

    def recv(self):
        raise self.exc


if __name__ == "__main__":
    unittest.main()
