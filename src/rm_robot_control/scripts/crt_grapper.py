#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import time
import threading
import minimalmodbus

from read_tactile_sensor_single import SerialReader

# ────────────── Modbus寄存器地址 ──────────────
ENABLE = 0x0100
POSITION_HIGH_8 = 0x0102
POSITION_LOW_8 = 0x0103
SPEED = 0x0104
FORCE = 0x0105
ACCELERATION = 0x0106
DEACCELERATION = 0x0107
MOTION_TRIGGER = 0x0108
RETURN_ZERO = 0x0402

# ────────────── 夹爪参数设定 ──────────────
PORT_GRAPPER = '/dev/ttyUSB0'     # 串口名称
BAUD = 115200             # 波特率
MAX_POSITION = 9000      # 最大位置值
GRIPPER_TRAVEL_CM = 9.75   # 行程长度（cm）

# ────────────── 触觉传感器参数设定 ──────────────
PORT_SENSOR = '/dev/ttyUSB1'     # 串口名称

class GripperController:
    def __init__(self, port=PORT_GRAPPER, slave_id=1):
        self.instrument = minimalmodbus.Instrument(port, slave_id)
        self.instrument.serial.baudrate = BAUD
        self.instrument.serial.timeout = 1
        self.lock = threading.Lock()

        self._init_gripper()

    def _init_gripper(self):
        self.enable()
        self.set_speed(100)
        self.set_force(50)
        self.set_acceleration(100)
        self.set_deacceleration(100)

    def enable(self):
        with self.lock:
            self.instrument.write_register(ENABLE, 1, functioncode=6)

    def set_position(self, position: int):
        """
        设置夹爪目标位置(0-9000), 0 为完全打开, 9000 为完全闭合。
        """
        if not 0 <= position <= MAX_POSITION:
            raise ValueError(f"位置应在 0 到 {MAX_POSITION} 之间")
        with self.lock:
            self.instrument.write_long(POSITION_HIGH_8, position)
            self.instrument.write_register(MOTION_TRIGGER, 1, functioncode=6)

    def set_position_cm(self, cm: float):
        """
        设置夹爪开口宽度（单位：厘米）, 0 为闭合, 9 为完全打开。
        """
        if not 0.0 <= cm <= GRIPPER_TRAVEL_CM:
            raise ValueError(f"开口宽度应在 0 到 {GRIPPER_TRAVEL_CM} cm 之间")
        position = int((1 - cm / GRIPPER_TRAVEL_CM) * MAX_POSITION)
        self.set_position(position)

    def set_speed(self, speed: int):
        with self.lock:
            self.instrument.write_register(SPEED, speed, functioncode=6)

    def set_force(self, force: int):
        with self.lock:
            self.instrument.write_register(FORCE, force, functioncode=6)

    def set_acceleration(self, acc: int):
        with self.lock:
            self.instrument.write_register(ACCELERATION, acc, functioncode=6)

    def set_deacceleration(self, deacc: int):
        with self.lock:
            self.instrument.write_register(DEACCELERATION, deacc, functioncode=6)

    def return_zero(self):
        with self.lock:
            self.instrument.write_register(RETURN_ZERO, 1, functioncode=6)

# 夹爪控制器
def grapper(pose): 
    # 初始化夹爪控制器
    gripper = GripperController()
    # 测试夹爪功能
    gripper.set_position(pose)

def grapper_tactile_sensor_control():
    gripper = GripperController()
    position = 0
    action_done = False
    STEP = 500                 # 每次闭合的步长
    PRESSURE_THRESHOLD = 600   # 压力阈值
    last_action_time = 0       # 节流控制时间戳
    ACTION_INTERVAL = 0.25     # 每次闭合动作的最小时间间隔（秒）

    def handle_matrix(mat):
        nonlocal action_done, position, last_action_time
        max_val = mat.max()
        print(f"[Console] Max pressure: {max_val}")

        current_time = time.time()

        if action_done:
            return
        
        if position > MAX_POSITION:
            position = MAX_POSITION
            print("夹爪已完全闭合")
            action_done = True

        if max_val < PRESSURE_THRESHOLD and (current_time - last_action_time) > ACTION_INTERVAL:
            position += STEP
            position = min(position, MAX_POSITION)
            gripper.set_position(position)
            last_action_time = current_time
        elif max_val >= PRESSURE_THRESHOLD:
            print("触觉值超阈，停止闭合并保持夹紧")
            action_done = True

    reader = SerialReader(PORT_SENSOR, 921600, callback=handle_matrix)
    reader.start()

    try:
        while not action_done:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[系统] 用户中断程序")
    finally:
        print("[系统] 停止串口读取")
        reader.stop()

# if __name__ == '__main__':
#     pose = 9000
#     grapper(pose)
#     time.sleep(3)
#     grapper_tactile_sensor_control()