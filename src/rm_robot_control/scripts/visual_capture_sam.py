import cv2
import numpy as np
import pyrealsense2 as rs

from scipy.spatial.transform import Rotation as R

from realsense import realsense
from single_frame_sam import SamPredict
from robot_control import RobotArmController

# 计算物体的朝向角度
def compute_angle_with_mask(mask):
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 初始化最大外接矩形
    min_rect = None
    max_area = 0

    for contour in contours:
        # 计算最小外接矩形
        center, (w, h), angle = cv2.minAreaRect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            min_rect = center, (w, h), angle

    # 获取最小外接矩形的信息
    center, (width, height), angle = min_rect

    if width > height:
        angle = -(90 - angle)
    else:
        angle = angle
    return angle, center

# 将旋转矩阵转换为齐次变换矩阵
def toHTM(rot):
    rot_HTM = np.eye(4)
    rot_HTM[:3, :3] = rot
    
    return rot_HTM


if __name__ == '__main__':

    # 垂直抓取流程：机械臂当前位姿 -> 相机Z轴竖直向下, X-Y平面与桌面平行 -> 夹爪与物体中心点对齐, X轴平行/垂直于物体 -> 抓取物体

    "-------------------------- 机械臂初始化 --------------------------"
    # 创建一个机器人手臂控制器实例, 并连接到机器人手臂
    robot_controller = RobotArmController("192.168.1.18", 8080, 3)
    
    "-------------------------- 相机Z轴竖直向下, X-Y平面与桌面平行 --------------------------"
    # 相机相对于机械臂末端的位姿
    rot_arm_camera = [[0.50117048, -0.8653173, 0.00735686],
                      [0.86448334, 0.5002712, -0.04896208],
                      [0.03868731, 0.03089823, 0.99877354]]
    T_arm_camera = toHTM(rot_arm_camera)
    T_arm_camera[:3, 3] = [ 0.08335923, -0.0377595, 0.02720036]
    # 相机初始位姿
    pose_init_camera = [0.1, -0.35, 0.1, np.pi/2, 0, np.pi/2]
    T_init_camera = np.eye(4)
    T_init_camera = robot_controller.pos2matrix(pose_init_camera)
    # 机械臂初始位姿
    T_init_arm = T_init_camera @ np.linalg.inv(T_arm_camera)
    pose_init_arm = np.zeros(6)
    pose_init_arm = robot_controller.matrix2pos(T_init_arm, 1)
    robot_controller.movej_p(pose_init_arm)
    
    "-------------------------- 识别物体类别、位置与朝向角 --------------------------"
    # 初始化 RealSense
    desired_serial = '244422300361' # 327522301155为工位相机，244422300361为realman左臂相机 
    realsense = realsense(desired_serial)
    # 获取一帧图像
    for _ in range(10): realsense.pipeline.wait_for_frames()
    print("准备从 Realsense 获取一帧图像... 按 's' 键保存当前帧并进行处理")
    while True:
        depth_intri, depth_frame, color_image, depth_image = realsense.get_aligned_images()
        cv2.imshow("RGB Frame (Press 's' to select)", color_image)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.destroyAllWindows()
            break
    
    # 初始化分割模型
    sam_checkpoint = r'/home/hh/realman_robotic_arm/src/rm_robot_control/model/sam/sam_vit_l.pth'
    sam_model_type = 'vit_l'
    sam_device = 'cuda'
    # 创建SamPredict类的实例
    model = SamPredict(image=color_image, checkpoint=sam_checkpoint, model_type=sam_model_type, model_device=sam_device, random_color=False)
    # 启动鼠标交互式标注
    model.interactive_point_selection()
    # 分割物体
    mask = model.get_result()
    
    "-------------------------- 夹爪与物体中心点对齐, X轴平行/垂直于物体 --------------------------"
    T_camera_obj_xyz = np.eye(4)
    T_camera_obj_rpy = np.eye(4)
    # 计算物体的朝向角度与中心点
    theta, center = compute_angle_with_mask(mask)
    center = (int(center[0]), int(center[1]))
    # 获取物体中心点的深度
    depth = depth_frame.get_distance(center[0], center[1])
    if depth > 0:
        # 夹爪与物体中心点对齐
        X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intri, [center[0], center[1]], depth)
        m = T_arm_camera[0, 3]
        T_camera_obj_xyz[0:3, 3] = [0, X + 0.01, Y - 0.09]
        T_base_obj_xyz = T_camera_obj_xyz @ T_init_arm
        pose_base_obj_xyz = robot_controller.matrix2pos(T_base_obj_xyz, 1)
        robot_controller.movej_p(pose_base_obj_xyz)
        # X轴平行/垂直于物体
        rot =  R.from_euler('z', np.deg2rad(theta)).as_matrix()
        T_camera_obj_rpy[0:3, 0:3] = rot
        T_base_obj_rpy = T_base_obj_xyz @ T_camera_obj_rpy
        pose_base_obj_rpy = robot_controller.matrix2pos(T_base_obj_rpy, 1)
        robot_controller.movej_p(pose_base_obj_rpy)
    else:
        print(f"array: depth为0, 无法获取物体中心点的3D坐标")
    
    # 与机械臂断连
    robot_controller.disconnect()