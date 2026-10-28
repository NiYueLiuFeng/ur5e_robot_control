import serial
serialPort = serial.Serial()

class dh_device(object) :

    def connect_device(self,portname, Baudrate) :
        ret = -1
        serialPort.port = portname
        serialPort.baudrate = Baudrate
        serialPort.bytesize = 8
        serialPort.parity = 'N'
        serialPort.stopbits = 1
        serialPort.xonxoff = False
        serialPort.rtscts = False
        serialPort.dsrdtr = False
        serialPort.timeout = 0.5
        serialPort.write_timeout = 0.5

        serialPort.open()
        if(serialPort.isOpen()) :
            print('Serial Open Success')
            ret = 0
        else :
            print('Serial Open Error')
            ret = -1
        return ret

    def disconnect_device() :
        if(serialPort.isOpen()) :
            serialPort.close()
        else :
            return

    def device_wrire(self, write_data) :
        write_lenght = 0
        if(serialPort.isOpen()) :
            write_lenght = serialPort.write(write_data)
            if(write_lenght == len(write_data)) :
                return write_lenght
            else :
                print('write error ! send_buff :',write_data)
                return 0;
        else :
            return -1

    def device_read(self, wlen) :
        if(serialPort.isOpen()) :
            # Modbus RTU is binary: 0x0a is data, not a line terminator.
            return serialPort.read(wlen)
        else :
            return b""

    def reset_input_buffer(self):
        if serialPort.isOpen():
            serialPort.reset_input_buffer()

    """description of class"""


