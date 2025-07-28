import cv2
import numpy as np
import pyrealsense2 as rs


class realsense:
    # 初始化 RealSense 摄像头
    def __init__(self, desired_serial):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(desired_serial)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    # 获取图像帧，返回深度和彩色数组
    def get_aligned_images(self):

        frames = self.pipeline.wait_for_frames()  # 等待获取图像帧
        aligned_frames = self.align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐
        depth_frame = aligned_frames.get_depth_frame()  # 获取深度帧
        color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧
        depth_image = np.asanyarray(depth_frame.get_data())  # 将深度帧转换为NumPy数组
        color_image = np.asanyarray(color_frame.get_data())  # 将彩色帧转化为numpy数组
        depth_intri = depth_frame.profile.as_video_stream_profile().intrinsics # 获取相机内参

        # 返回深度内参、对齐深度帧、彩色图像
        return depth_intri, depth_frame, color_image, depth_image

    # 将像素坐标 (x, y) 和深度帧转为三维坐标（以相机为参考）
    def get_3d_position(self, x, y, depth_frame, depth_intrinsics):
        depth = depth_frame.get_distance(x, y)  # 单位为米
        if depth == 0:
            return None
        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
        
        return point_3d  # 返回 [X, Y, Z]，单位为米


if __name__ == '__main__':
    
    # 检测 RealSense 设备, 获取设备码
    ctx = rs.context()
    print("--" * 50)
    for device in ctx.devices:
        print("Found device:", device.get_info(rs.camera_info.name), 
            "Serial number:", device.get_info(rs.camera_info.serial_number))

    # 初始化 RealSenses
    desired_serial = '244422300361'
    realsense = realsense(desired_serial)

    while True:
        depth_intri, depth_frame, color_image, depth_image = realsense.get_aligned_images()
        cv2.imshow('realsense', color_image)
        if cv2.waitKey(100) == 27:  # ESC退出
            break
