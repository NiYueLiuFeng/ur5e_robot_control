import os
import sys
import glob 
import time

import numpy as np

from  dh_driver.dh_modbus_gripper import dh_modbus_gripper

class GripperControl:
    def __init__(self):
        self.m_gripper = dh_modbus_gripper()
        port='/dev/ttyUSB0'
        baudrate=115200
        initstate = 0
        g_state = 0
        force = 100
        speed = 100

        self.m_gripper.open(port,baudrate)
        self.m_gripper.Initialization()
        print('Send grip init')

        while(initstate != 1) :
            initstate = self.m_gripper.GetInitState()
            time.sleep(0.2)
            
        self.m_gripper.SetTargetForce(force)
        self.m_gripper.SetTargetSpeed(speed)


    def step_grip(self, grasp_command):
        """
        g: 0 打开  1 关闭
        """
        # Set gripper grasp
        g = self.get_gripper_position() / 1000
        ng = np.clip(g + grasp_command, 0.0, 1.0)
        grapper_pos = int(ng * 1000)            
        self.m_gripper.SetTargetPosition(grapper_pos)

    def get_gripper_position(self):
        """
        获取夹爪的状态
        """
        g_state = self.m_gripper.GetCurrentPosition()
        return g_state

    def get_norm_gripper_position(self):
        return  1 if self.m_gripper.GetCurrentPosition() < 500 else 0

    def disconnect(self):
        self.m_gripper.close()