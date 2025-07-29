#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read_tactile_vis_fixed_log.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
• PyQt5 + PyQtGraph 实时显示 5 条 daisy-chain（10 片 12×8 阵列）压感热力图  
• 同步将每完整帧写入 JSON Lines (tactile_stream.jsonl)  
• 通过 --freq/-f 命令行参数控制更新频率（FPS）

示例:
    python read_tactile_vis_fixed_log.py           # 默认 20 FPS
    python read_tactile_vis_fixed_log.py -f 15     # 15 FPS
"""

import sys
import time
import struct
import json
import argparse                             # ← 新增

import numpy as np
import serial

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# ────────── 可通过命令行覆盖的默认参数 ──────────
PORT     = '/dev/ttyUSB0'
BAUD     = 921_600
LIVE_IDS = [1, 2, 3, 4, 5]
FPS_DEF  = 20.0                            # 默认 20 帧/秒
LOG_FILE = 'tactile_stream.jsonl'
# ─────────────────────────────────────────

# ────────── 协议常量 ─────────────────────
HEAD, TAIL = b'\x3C\x3C', b'\x3E\x3E'
CHAN_DATA  = 0x02
CHAN_CONF  = 0x06
GET_FLAGS  = 0xA5
PUT_FLAGS  = 0x00
PAY_GET    = b'\x01'
PAY_ENABLE = b'\x01\x01'
# ─────────────────────────────────────────

from typing import List, Optional


def crc16(buf: bytes) -> bytes:
    crc = 0
    for b in buf:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if (crc & 1) else (crc >> 1)
    return struct.pack('<H', crc)


def pack(dev_id: int, channel: int, flags: int, payload: bytes) -> bytes:
    pkt_id = (dev_id << 4) | channel
    return (
        HEAD
        + bytes([pkt_id, flags])
        + struct.pack('<H', len(payload))
        + payload
        + crc16(payload)
        + TAIL
    )


def cmd_enable(dev_id: int) -> bytes:
    return pack(dev_id, CHAN_CONF, PUT_FLAGS, PAY_ENABLE)


def cmd_get(dev_id: int) -> bytes:
    return pack(dev_id, CHAN_DATA, GET_FLAGS, PAY_GET)


def read_packet(ser: serial.Serial, timeout: float = 0.02) -> Optional[bytes]:
    ser.timeout = timeout
    hdr = ser.read(6)
    if len(hdr) < 6 or hdr[:2] != HEAD:
        return None
    plen = struct.unpack_from('<H', hdr, 4)[0]
    body = ser.read(plen + 4)
    if len(body) != plen + 4 or body[-2:] != TAIL:
        return None
    return hdr + body


def parse_payload(pay: bytes) -> Optional[List[np.ndarray]]:
    if len(pay) != 391 or pay[2:4] != b'\x08\x0c':
        return None
    blk1 = pay[4:196]
    blk2 = pay[199:391]
    if len(blk1) != 192 or len(blk2) != 192:
        return None
    m1 = np.frombuffer(blk1, dtype='<u2').reshape(12, 8)
    m2 = np.frombuffer(blk2, dtype='<u2').reshape(12, 8)
    return [m1, m2]


def main():
    # ─────── 解析命令行 ───────────────────
    parser = argparse.ArgumentParser(
        description="Pentachain tactile viewer/logger")
    parser.add_argument('-f', '--freq', type=float, default=FPS_DEF,
                        help=f'更新帧率 (Hz)，默认 {FPS_DEF}')
    args = parser.parse_args()
    fps = max(args.freq, 0.1)              # 防止 0 或负数
    PRINT_GAP = 1.0 / fps

    # ─────── 打开串口 ─────────────────────
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1.0)
    except serial.SerialException as e:
        sys.exit(f"串口打开失败: {e}")
    print(f"Opened {PORT} @ {BAUD}  |  {fps:.1f} FPS")

    # ─────── 使能节点 ─────────────────────
    for pid in LIVE_IDS:
        ser.write(cmd_enable(pid))
        time.sleep(0.02)
    time.sleep(0.1)

    print(f"轮询节点 {LIVE_IDS} → 实时显示并记录至 {LOG_FILE}\n"
          f"关闭窗口或 Ctrl+C 退出")

    # ─────── 初始化 Qt & 图形 ─────────────
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)

    win = pg.GraphicsLayoutWidget(show=True, title="Pentachain 实时热力图")
    win.resize(800, 1200)

    img_items: List[pg.ImageItem] = []
    VMIN, VMAX = 0, 1023
    lut = pg.colormap.get('viridis').getLookupTable(nPts=256)

    for row in range(5):
        for col in range(2):
            p = win.addPlot(row=row, col=col)
            p.hideAxis('bottom')
            p.hideAxis('left')
            p.setAspectLocked(True)
            dummy = np.zeros((8, 12), dtype=np.uint16)
            img = pg.ImageItem(dummy)
            img.setLookupTable(lut)
            img.setLevels([VMIN, VMAX])
            p.addItem(img)
            p.setTitle(f"ID={LIVE_IDS[row]}  阵列#{(col%2)+1}", size="8pt")
            img_items.append(img)

    # ─────── 打开 JSONL 文件 ──────────────
    try:
        log_f = open(LOG_FILE, "a", buffering=1, encoding="utf-8")
    except OSError as e:
        sys.exit(f"无法打开日志文件 {LOG_FILE}: {e}")

    # ─────── 定时器回调 ───────────────────
    def update():
        mats_buf: List[np.ndarray] = []
        idx = 0
        for pid in LIVE_IDS:
            ser.write(cmd_get(pid))
            pkt = read_packet(ser)
            if pkt:
                plen = struct.unpack_from('<H', pkt, 4)[0]
                payload = pkt[6:6+plen]
                mats = parse_payload(payload)
                if mats:
                    for mat in mats:
                        arr = mat.T
                        img_items[idx].setImage(arr, levels=(VMIN, VMAX))
                        idx += 1
                        mats_buf.append(arr.astype(int))
                    continue
            # 若失败：补零
            for _ in range(2):
                img_items[idx].setImage(np.zeros((8, 12),
                                                 dtype=np.uint16),
                                         levels=(VMIN, VMAX))
                idx += 1
        if len(mats_buf) == 10:
            json.dump({"ts": time.time(),
                       "mats": [m.tolist() for m in mats_buf]}, log_f)
            log_f.write("\n")

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(int(PRINT_GAP * 1000))     # ms

    # ─────── 主循环 ──────────────────────
    try:
        sys.exit(app.exec_())
    finally:
        if ser.is_open:
            ser.close()
        log_f.close()
        print("串口已关闭，日志文件写入完成，程序退出。")


if __name__ == '__main__':
    main()
