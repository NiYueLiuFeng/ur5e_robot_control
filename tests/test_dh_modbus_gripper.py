import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import serial


def load_sdk_modules():
    sdk_root = Path(__file__).resolve().parents[1] / "src" / "dh_gripper"
    modules = []
    for name in ("dh_device", "dh_modbus_gripper"):
        spec = importlib.util.spec_from_file_location(name, sdk_root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"dh_device": modules[0]} if modules else {}):
            spec.loader.exec_module(module)
        modules.append(module)
    return modules


class DHModbusGripperTest(unittest.TestCase):
    def setUp(self):
        self.device_module, self.driver_module = load_sdk_modules()
        self.device = Mock()
        self.device.device_wrire.return_value = 8
        self.driver_module.m_device = self.device
        self.gripper = self.driver_module.dh_modbus_gripper()

    def test_reads_position_from_valid_crc_checked_response(self):
        self.device.device_read.return_value = bytes.fromhex("01 03 02 03 e8 b8 fa")

        self.assertEqual(self.gripper.GetCurrentPosition(), 1000)
        request = bytes(self.device.device_wrire.call_args.args[0])
        self.assertEqual(request, bytes.fromhex("01 03 02 02 00 01 24 72"))

    def test_retries_misaligned_response_instead_of_returning_770(self):
        self.device.device_read.side_effect = [
            bytes.fromhex("aa bb 01 03 02 03 e8"),
            bytes.fromhex("01 03 02 03 e8 b8 fa"),
        ]

        self.assertEqual(self.gripper.GetCurrentPosition(), 1000)
        self.assertEqual(self.device.device_read.call_count, 2)
        self.assertEqual(self.device.reset_input_buffer.call_count, 2)

    def test_accepts_770_when_it_is_actual_crc_checked_position_data(self):
        self.device.device_read.return_value = bytes.fromhex("01 03 02 03 02 39 75")

        self.assertEqual(self.gripper.GetCurrentPosition(), 770)

    def test_rejects_bad_crc_with_register_and_packet_in_error(self):
        self.device.device_read.return_value = bytes.fromhex("01 03 02 03 e8 00 00")

        with self.assertRaisesRegex(RuntimeError, "0x0202.*01 03 02 03 e8 00 00"):
            self.gripper.GetCurrentPosition()

    def test_short_response_raises_communication_error(self):
        self.device.device_read.return_value = b""

        with self.assertRaisesRegex(RuntimeError, "0x0202"):
            self.gripper.GetCurrentPosition()

    def test_write_retries_response_for_wrong_register(self):
        self.device.device_read.side_effect = [
            bytes.fromhex("01 06 01 04 03 e8 c9 49"),
            bytes.fromhex("01 06 01 03 03 e8 78 88"),
        ]

        self.assertTrue(self.gripper.WriteRegisterFunc(0x0103, 1000))
        self.assertEqual(self.device.device_read.call_count, 2)

    def test_serial_read_keeps_newline_byte_inside_binary_response(self):
        response = bytes.fromhex("01 03 02 00 0a 38 43")
        with serial.serial_for_url("loop://", timeout=0.1) as port:
            port.write(response)
            with patch.object(self.device_module, "serialPort", port):
                self.assertEqual(
                    self.device_module.dh_device().device_read(7), response
                )


if __name__ == "__main__":
    unittest.main()
