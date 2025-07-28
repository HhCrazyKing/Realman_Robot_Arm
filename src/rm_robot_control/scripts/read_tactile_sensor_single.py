#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import time
import serial
import struct
import numpy as np
import scipy.ndimage
import pyqtgraph as pg

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication

PORT_SENSOR = '/dev/ttyUSB1'
HEAD, TAIL = b'\x3C\x3C', b'\x3E\x3E'

# ─────────────── 串口数据读取模块 ────────────────
class SerialReader(QtCore.QThread):
    new_matrix = QtCore.pyqtSignal(object)

    def __init__(self, port=PORT_SENSOR, baudrate=921600, timeout=0.01, callback=None):
        super().__init__()
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.running = True
        self.buffer = bytearray()
        self.callback = callback

    # 将接收到的字节流解析为矩阵
    def run(self):
        while self.running:
            try:
                data = self.ser.read(1024)
                if not self.running:
                    break
                if data:
                    self.buffer += data
                    while True:
                        pkt = self.extract_frame()
                        if pkt:
                            plen = struct.unpack_from('<H', pkt, 4)[0]
                            payload = pkt[6:6 + plen]
                            matrix = self.parse_payload(payload)
                            if matrix is not None:
                                if self.callback:
                                    self.callback(matrix)
                                self.new_matrix.emit(matrix)
                        else:
                            break
            except Exception as e:
                print("串口读取异常:", e)

    # 提取完整的帧数据
    def extract_frame(self):
        while True:
            start = self.buffer.find(HEAD)
            end = self.buffer.find(TAIL, start + 2)
            if start != -1 and end != -1 and end > start:
                frame = bytes(self.buffer[start:end + 2])
                del self.buffer[:end + 2]
                return frame
            else:
                break
        return None

    # 解析负载数据为矩阵, 触觉传感器数据格式为 12x8 的矩阵，每个元素为 2 字节无符号整数
    def parse_payload(self, pay: bytes):
        if len(pay) < 196 or pay[2:4] != b'\x08\x0c':
            return None
        blk1 = pay[4:196]
        if len(blk1) != 192:
            return None
        m1 = np.frombuffer(blk1, dtype='<u2').reshape(12, 8)
        return m1

    # 停止线程并关闭串口
    def stop(self):
        print("[系统] 正在关闭串口线程...")
        self.running = False
        self.wait()  # 等待 run() 正常退出
        if self.ser.is_open:
            self.ser.close()
        print("[系统] 串口线程已停止")



# ─────────────── 可视化模块 ────────────────
class TactileVisualizer(QtWidgets.QMainWindow):
    def __init__(self, port=PORT_SENSOR, baudrate=921600, zoom_factor=8):
        super().__init__()
        self.zoom_factor = zoom_factor
        self.raw_matrix = None
        self.upsampled_matrix = None

        # 图像窗口
        self.img = pg.ImageView()
        self.setCentralWidget(self.img)
        self.setWindowTitle("触觉热力图")
        self.setGeometry(100, 100, 800, 600)

        self.img.ui.histogram.hide()
        self.img.ui.roiBtn.hide()
        self.img.ui.menuBtn.hide()
        self.img.getView().setBackgroundColor('w')
        colormap = pg.colormap.get('inferno')
        self.img.setColorMap(colormap)
        self.img.setLevels(0, 1024)

        # 串口读取器
        self.reader = SerialReader(port, baudrate, callback=self.on_matrix)
        self.reader.new_matrix.connect(self.update_heatmap)
        self.reader.start()

    @QtCore.pyqtSlot(np.ndarray)
    def update_heatmap(self, mat):
        self.raw_matrix = mat
        self.upsampled_matrix = scipy.ndimage.zoom(mat, self.zoom_factor, order=3)
        self.img.setImage(self.upsampled_matrix.T, autoLevels=False)

    def on_matrix(self, mat):
        # 可在此做分析逻辑，例如超过阈值报警
        max_val = mat.max()
        max_pos = np.unravel_index(np.argmax(mat), mat.shape)
        print(f"[GUI] Max pressure: {max_val} at {max_pos}")

    def get_raw_matrix(self):
        return self.raw_matrix

    def get_heatmap(self):
        return self.upsampled_matrix

    def closeEvent(self, event):
        self.reader.stop()
        event.accept()

# gui函数用于启动图形界面
def tactile_sensor_gui():
    app = QApplication(sys.argv)
    vis = TactileVisualizer(port=PORT_SENSOR)
    vis.show()
    sys.exit(app.exec_())

# 仅数据处理，不启动图形界面
def tactile_sensor_no_gui():
    def handle_matrix(mat):
        max_val = mat.max()
        max_pos = np.unravel_index(np.argmax(mat), mat.shape)
        print(f"[Console] Max pressure: {max_val} at {max_pos}")
        # 可加入触发条件逻辑，例如：
        if max_val > 600:
            print("[警告] 触觉值超阈，执行控制动作...")

    reader = SerialReader(PORT_SENSOR, 921600, callback=handle_matrix)
    reader.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n终止串口读取")
        reader.stop()

# ─────────────── 主入口 ────────────────
if __name__ == '__main__':
    tactile_sensor_gui()