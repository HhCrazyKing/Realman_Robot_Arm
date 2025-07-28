import os
import sys
import cv2
import random
import numpy as np
import pyrealsense2 as rs

from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realsense import realsense
from rm_reference_code.SAM.segment_anything import sam_model_registry, SamPredictor

class SamYoloPredict: 
    # 图片（any）、模型（any）
    def __init__(self, image):
        self.image = image
    
    # 计算物体的平面朝向角度与中心点坐标
    def compute_angle_with_mask(self, mask, depth_frame, depth_intri):

        # 将 mask 转换为 uint8 类型并找到轮廓
        mask_uint8 = (mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 初始化最大外接矩形
        min_rect = None
        max_area = 0
        # 遍历所有轮廓，找到面积最大的轮廓
        for contour in contours:
            # 计算最小外接矩形
            center, (w, h), theta = cv2.minAreaRect(contour)
            area = w * h
            if area > max_area:
                max_area = area
                min_rect = center, (w, h), theta
        # 获取最小外接矩形的信息
        center, (width, height), theta = min_rect
        # 确保角度在 -90 到 90 度之间
        if width > height:
            theta = -(90 - theta)
        else:
            theta = theta
        # 从 mask 中找一个最近的非零深度点
        x_center, y_center = int(round(center[0])), int(round(center[1]))
        search_radius = 15
        if depth_frame is not None:
            depth = depth_frame.get_distance(x_center, y_center)
            if depth == 0:
                mask_indices = np.argwhere(mask)
                for radius in range(1, search_radius + 1):
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            x, y = x_center + dx, y_center + dy
                            if (0 <= y < mask.shape[0]) and (0 <= x < mask.shape[1]):
                                if mask[y, x] > 0:
                                    d = depth_frame.get_distance(x, y)
                                    if d > 0:
                                        x, y, z = rs.rs2_deproject_pixel_to_point(depth_intri, [x, y], d)
                                        return theta, (x, y, z)
                return theta, (x_center, y_center, 0.0)
            x_center, y_center, depth_center = rs.rs2_deproject_pixel_to_point(depth_intri, [x_center, y_center], depth)
            return theta, (x_center, y_center, depth_center)
    
    # 物体的六维位姿估计
    def estimate_pose_from_mask(self, mask, depth_frame, depth_intrin):

        # 将 mask 转换为二值图像
        mask = mask.astype(np.uint8)
        indices = np.argwhere(mask)
        # 获取mask中所有非零点的深度值, 三维点云
        points_3d = []
        for v, u in indices:
            depth = depth_frame.get_distance(u, v)
            if depth > 0:
                X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], depth)
                points_3d.append([X, Y, Z])
        points_3d = np.array(points_3d)
        # 平面拟合 -> PCA 主方向
        if len(points_3d) >= 3:
            # 使用 SVD 进行主成分分析
            centroid = np.mean(points_3d, axis=0)
            centered = points_3d - centroid
            _, _, vh = np.linalg.svd(centered)
            # vh 的行向量是主成分方向
            z_axis = -vh[2]  # 假设 mask 朝向为 Z 轴（最小变化方向）
            x_axis = vh[0]  # 第二主方向
            y_axis = np.cross(z_axis, x_axis)  # 保持右手系
            # 单位化
            x_axis /= np.linalg.norm(x_axis)
            y_axis /= np.linalg.norm(y_axis)
            z_axis /= np.linalg.norm(z_axis)
            # 构建旋转矩阵
            rot_mat = np.vstack((x_axis, y_axis, z_axis)).T

            return centroid, rot_mat
        else:
            return None, None

    # 绘制物体的rpy姿态坐标系
    def draw_pose(self, image, depth_intri, pose_xyz, rot_mat):

        # 可视化姿态坐标系
        scale = 0.05  # 坐标轴长度（单位：米）
        # 原点（物体中心点）
        origin = pose_xyz
        # 坐标轴单位向量
        x_axis_3d = (rot_mat @ np.array([scale, 0, 0])) + origin
        y_axis_3d = (rot_mat @ np.array([0, scale, 0])) + origin
        z_axis_3d = (rot_mat @ np.array([0, 0, scale])) + origin
        # 坐标系投影到图像上
        origin_2d = rs.rs2_project_point_to_pixel(depth_intri, origin.tolist())
        x_2d = rs.rs2_project_point_to_pixel(depth_intri, x_axis_3d.tolist())
        y_2d = rs.rs2_project_point_to_pixel(depth_intri, y_axis_3d.tolist())
        z_2d = rs.rs2_project_point_to_pixel(depth_intri, z_axis_3d.tolist())
        # 绘制坐标轴（X-红，Y-绿，Z-蓝）
        origin_2d = tuple(map(int, origin_2d))
        cv2.arrowedLine(image, origin_2d, tuple(map(int, x_2d)), (0, 0, 255), 2, tipLength=0.3)  # X
        cv2.arrowedLine(image, origin_2d, tuple(map(int, y_2d)), (0, 255, 0), 2, tipLength=0.3)  # Y
        cv2.arrowedLine(image, origin_2d, tuple(map(int, z_2d)), (255, 0, 0), 2, tipLength=0.3)  # Z

        return image

    # 检测并分割物体
    def identification_segmentation(self, depth_intri, depth_frame, yolo, predictor, conf_threshold=0.4):

        # 识别模型
        COCO_CLASSES = yolo.model.names
        # 识别结果
        results = yolo(self.image)[0]                # results为YOLO模型的输出
        boxes = results.boxes.xyxy.cpu().numpy()     # 获取检测框坐标
        scores = results.boxes.conf.cpu().numpy()    # 获取置信度分数
        class_ids = results.boxes.cls.cpu().numpy()  # 获取类别ID
        print(f"检测到 {len(boxes)} 个物体")

        # 分割模型
        predictor.set_image(self.image)
        
        # 变量初始化
        count = 0                        # 物体计数
        label_count = {}               # 用于存储物体类别计数
        all_object_info = []             # 存储所有物体信息
        vis_image = self.image.copy()    # 可视化图像

        # 给检测到的物体添加边框、标签与三维位置信息
        for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):

            # 过滤低置信度的检测结果
            if score < conf_threshold:
                continue

            # 计算中心点坐标
            x1, y1, x2, y2 = map(int, box)                         # 转换为整数
            point = np.array([[(x1 + x2) // 2, (y1 + y2) // 2]])   # 中心点坐标
            label = np.array([1])                                  # 标签为1，表示前景点
            
            # 获取分割结果
            masks, scores_sam, _ = predictor.predict(
                point_coords=point,
                point_labels=label,
                multimask_output=True)
            best_mask = masks[np.argmax(scores_sam)]

            color = [random.randint(0, 255) for _ in range(3)]
            # 初始化或更新物体类别计数
            label_text = COCO_CLASSES[int(cls_id)]
            if label_text not in label_count:
                label_count[label_text] = 1
            else:
                label_count[label_text] += 1
            # 添加编号后的新标签
            label_numbered = f"{label_text} {label_count[label_text]}"
            # 在图像上绘制检测框和标签
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_image, label_numbered, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, thickness=2)

            "-------------------------- 根据物品的姿态估计夹爪的朝向 --------------------------"
            # 计算物体的六维位姿与尺寸
            theta, center = self.compute_angle_with_mask(best_mask, depth_frame, depth_intri)
            pose_xyz, rot_mat = self.estimate_pose_from_mask(best_mask, depth_frame, depth_intri)
            if pose_xyz is not None:
                euler = R.from_matrix(rot_mat).as_euler('xyz', degrees=True)
                all_object_info.append({
                    "label": label_numbered,
                    "XYZ": pose_xyz.tolist(),
                    "rotation_matrix": rot_mat.tolist(),
                    "euler_angles": euler.tolist(),
                    "theta": theta,
                    "center": center
                })

                print(f"物体 {count + 1}: 标签 = {label_numbered}, 置信度 = {score:.3f}, 六维位姿 = ({pose_xyz[0]:.3f}, {pose_xyz[1]:.3f}, {pose_xyz[2]:.3f}, {euler[0]:.3f}, {euler[1]:.3f}, {euler[2]:.3f}), "
                      f"朝向角 = ({theta:.3f})°, 中心点 = ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
                vis_image = self.draw_pose(vis_image, depth_intri, pose_xyz, rot_mat)
            else:
                print(f"物体 {count + 1}: 标签 = {label_numbered}: 无法估计姿态")

            count += 1

        return all_object_info, vis_image

if __name__ == '__main__':

    # 初始化 RealSense
    desired_serial = '327522301155'
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
    all_object_info, image = model.identification_segmentation(depth_intri=depth_intri, depth_frame=depth_frame, yolo=yolo, predictor=predictor)

    cv2.imshow("Segmented Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        