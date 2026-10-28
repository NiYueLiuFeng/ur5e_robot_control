import sys

sys.path.append("dh_gripper/dh_driver")
from  dh_device  import dh_device

m_device = dh_device()

class dh_modbus_gripper(object):
    gripper_ID = 0x01

    def CRC16(self,nData, wLength) :
        if nData==0x00:
            return 0x0000
        wCRCWord=0xFFFF
        poly=0xA001
        for num in range(wLength):
            date = nData[num]
            wCRCWord = (date & 0xFF)^ wCRCWord
            for bit in range(8) :
                if(wCRCWord&0x01)!=0:
                    wCRCWord>>=1
                    wCRCWord^= poly
                else:
                    wCRCWord>>=1
        return wCRCWord

    def open(self,PortName,BaudRate) :
        ret = 0
        ret = m_device.connect_device(PortName, BaudRate)
        if(ret < 0) :
            print('open failed')
            return ret
        else :
            print('open successful')
            return ret

    def close() :
        m_device.disconnect_device()

    def WriteRegisterFunc(self,index, value) :
        send_buf = [0,0,0,0,0,0,0,0]
        send_buf[0] = self.gripper_ID
        send_buf[1] = 0x06
        send_buf[2] = (index >> 8) & 0xFF
        send_buf[3] = index & 0xFF
        send_buf[4] = (value >> 8) & 0xFF
        send_buf[5] = value & 0xFF

        crc = self.CRC16(send_buf,len(send_buf)-2)
        send_buf[6] = crc & 0xFF
        send_buf[7] = (crc >> 8) & 0xFF

        request = bytes(send_buf)
        # Function 0x06 must echo the complete request, including its CRC.
        self._exchange(request, 8, lambda response: response == request)
        return True

    def ReadRegisterFunc(self,index) :
        send_buf = [0,0,0,0,0,0,0,0]
        send_buf[0] = self.gripper_ID
        send_buf[1] = 0x03
        send_buf[2] = (index >> 8) & 0xFF
        send_buf[3] = index & 0xFF
        send_buf[4] = 0x00
        send_buf[5] = 0x01

        crc = self.CRC16(send_buf,len(send_buf)-2)
        send_buf[6] = crc & 0xFF
        send_buf[7] = (crc >> 8) & 0xFF

        def valid_response(response):
            return (
                len(response) == 7
                and response[:3] == bytes((self.gripper_ID, 0x03, 0x02))
                and self.CRC16(response, 5)
                == int.from_bytes(response[-2:], "little")
            )

        response = self._exchange(bytes(send_buf), 7, valid_response)
        return int.from_bytes(response[3:5], "big")

    def _exchange(self, request, response_length, valid_response):
        last_response = b""
        for _ in range(4):
            # Discard stale/partial replies before retrying a transaction.
            m_device.reset_input_buffer()
            if m_device.device_wrire(request) != len(request):
                continue
            last_response = bytes(m_device.device_read(response_length))
            if valid_response(last_response):
                return last_response

        register = int.from_bytes(request[2:4], "big")
        raise RuntimeError(
            f"Modbus transaction failed for register 0x{register:04x}; "
            f"last response: {last_response.hex(' ') or '<empty>'}"
        )

    def Initialization(self) :
        self.WriteRegisterFunc(0x0100,0xA5)

    def SetTargetPosition(self,refpos) :
        self.WriteRegisterFunc(0x0103,refpos);

    def SetTargetForce(self,force) :
        self.WriteRegisterFunc(0x0101,force);

    def SetTargetSpeed(self,speed) :
        self.WriteRegisterFunc(0x0104,speed);

    def GetCurrentPosition(self) :
        return self.ReadRegisterFunc(0x0202);

    def GetCurrentTargetForce(self) :
        return self.ReadRegisterFunc(0x0101);

    def GetCurrentTargetSpeed(self) :
        return self.ReadRegisterFunc(0x0104);

    def GetInitState(self) :
        return self.ReadRegisterFunc(0x0200);

    def GetGripState(self) :
        return self.ReadRegisterFunc(0x0201);

    """description of class"""


