import os
import sys
import cv2
import time
import numpy as np

from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realsense import realsense
from crt_grapper import GripperController
from robot_control import RobotArmController
from single_frame_sam_yolo import SamYoloPredict
from rm_reference_code.SAM.segment_anything import sam_model_registry, SamPredictor

# 将旋转矩阵转换为齐次变换矩阵
def toHTM(rot):
    rot_HTM = np.eye(4)
    rot_HTM[:3, :3] = rot
    
    return rot_HTM

# 抓取物体
def capture(obj, gripper_pose):
    # 夹爪与物体中心点对齐
    pose_xyz = obj['XYZ']
    # T_camera_obj_xyz[0:3, 3] = [0, pose_xyz[1] - T_arm_camera[0, 3], -pose_xyz[0] - T_arm_camera[2, 3]]   # 机械臂在正前方，相机朝向正前方
    T_arm_obj_xyz[0:3, 3] = [0, -pose_xyz[1] + T_arm_camera[0, 3], pose_xyz[0] + T_arm_camera[1, 3]]   # 机械臂在正后方，相机朝向正后方
    T_base_obj_xyz = T_arm_obj_xyz @ T_init_arm
    pose_base_obj_xyz = robot_controller.matrix2pos(T_base_obj_xyz, 1)
    robot_controller.movej_p(pose_base_obj_xyz)
    # 夹爪与物体朝向对齐
    theta = obj['theta']
    rot = R.from_euler('z', np.deg2rad(theta)).as_matrix()
    T_arm_obj_rpy[0:3, 0:3] = rot
    T_base_obj_rpy = T_base_obj_xyz @ T_arm_obj_rpy
    pose_base_obj_rpy = robot_controller.matrix2pos(T_base_obj_rpy, 1)
    robot_controller.movej_p(pose_base_obj_rpy)
    # 向下移动夹爪并抓取物体
    pose_base_grapper_down = pose_base_obj_rpy.copy()
    down_distance = pose_xyz[2] - 0.22
    pose_base_grapper_down[0] += down_distance
    print(pose_base_grapper_down[0])
    robot_controller.movej_p(pose_base_grapper_down)
    gripper.set_position(gripper_pose)
    time.sleep(1.5)
    # 抬起夹爪
    pose_base_grapper_up = pose_base_grapper_down.copy()
    up = 0.20
    pose_base_grapper_up[0] -= up
    robot_controller.movej_p(pose_base_grapper_up)

# 输入指定的抓取目标
def input_target_labels():
    while True:
        input_str = input("请输入目标物体名称列表 (例如: cell phone, cup, book): ")
        target_labels = [s.strip().lower() for s in input_str.split(',') if s.strip()]
        # 找出未匹配到的标签
        unmatched = [label for label in target_labels if label not in label_to_object]
        if unmatched:
            print(f"⚠️ 以下物体未在当前视野中识别到：{', '.join(unmatched)}")
            print("请重新输入完整的物体列表（所有目标必须存在）")
        elif not target_labels:
            print("⚠️ 输入为空，请重新输入。")
        else:
            break
    
    return target_labels

# 输入指定的放置位置
def input_specified_location(prompt_msg="请输入指定放置位置（例如: box): "):
    while True:
        label = input(prompt_msg).strip().lower()
        if label not in label_to_object:
            print(f"⚠️ 物体 “{label}” 不在当前识别列表中，请重新输入。")
        else:
            return label


if __name__ == '__main__':
    
    try:
        # 垂直抓取流程：机械臂当前位姿 -> 相机Z轴竖直向下, X-Y平面与桌面平行 -> 夹爪与物体中心点对齐, X轴平行/垂直于物体 -> 抓取物体

        "-------------------------- 机械臂初始化 --------------------------"
        # 创建一个机器人手臂控制器实例, 并连接到机器人手臂
        robot_controller = RobotArmController("192.168.1.18", 8080, 3)
        # 获取机械臂的基本信息
        robot_controller.get_arm_software_info()
        # 初始化夹爪控制器
        gripper = GripperController()
        gripper.set_position(0)
        
        "-------------------------- 相机Z轴竖直向下, X-Y平面与桌面平行 --------------------------"
        # 相机相对于机械臂末端的位姿
        rot_arm_camera = [[-0.01451765, -0.99967178, -0.02110838],
                          [ 0.99987844, -0.01439412, -0.00599222],
                          [ 0.00568642, -0.02119281,  0.99975924]]
        T_arm_camera = toHTM(rot_arm_camera)
        T_arm_camera[:3, 3] =  [0.08424036, 0.00551616, 0.04064065]
        # 机械臂初始位姿
        pose_init_arm = [0.1, 0.35, 0.1, np.pi/2, 0, np.pi/2]
        robot_controller.movej_p(pose_init_arm)
        # 相机初始位姿
        T_init_arm = robot_controller.pos2matrix(pose_init_arm)
        
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
        
        # 初始化识别模型
        yolo = YOLO("/home/hh/realman_robotic_arm/src/rm_robot_control/model/yolo/yolo11x.pt")
        # 初始化分割模型
        sam_checkpoint = r'/home/hh/realman_robotic_arm/src/rm_robot_control/model/sam/sam_vit_l.pth'
        sam_model_type = 'vit_l'
        sam_device = 'cuda'
        model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        model.to(device=sam_device)
        predictor = SamPredictor(model)
        # 创建SamPredict类的实例
        model = SamYoloPredict(image=color_image)
        # 检测并分割物体
        all_object_info, vis_image = model.identification_segmentation(depth_intri=depth_intri, depth_frame=depth_frame, yolo=yolo, predictor=predictor)
        
        "-------------------------- 夹爪与物体中心点对齐, X轴平行/垂直于物体 --------------------------"
        # 变量初始化
        T_arm_obj_xyz = np.eye(4)
        T_arm_obj_rpy = np.eye(4)
        # 创建一个字典，方便快速查找
        label_to_object = {obj['label'].lower(): obj for obj in all_object_info}
        # 允许用户输入多个目标标签，用逗号分隔
        target_labels = input_target_labels()
        specified_location = input_specified_location()
        # 遍历目标标签，按顺序执行移动操作
        for target_label in target_labels:
            # 抓取指定物体
            print(f"正在处理物体：{target_label}")
            target = label_to_object[target_label]
            capture(target, 9000)
            # 将物体放置在指定位置
            location = label_to_object[specified_location]
            capture(location, 0)

            time.sleep(1)

        # 机械臂回到初始位姿
        robot_controller.movej_p(pose_init_arm)
        # 与机械臂断连
        robot_controller.disconnect()

    except SystemExit:
        print("⚠️ 程序因机械臂移动失败而退出。")