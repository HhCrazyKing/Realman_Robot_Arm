import os
import sys
import cv2
import numpy as np
import supervision as sv
import pyrealsense2 as rs

from datetime import datetime
from collections import defaultdict
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task
from scipy.spatial.transform import Rotation as R
from dds_cloudapi_sdk.image_resizer import image_to_base64

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realsense import realsense
from rm_reference_code.DINO_X.rle_util import rle_to_array


class DINOXObjectDetector:
    def __init__(self):
        # 超参数初始化
        self.api_token = "0768f2df00d338f57200ac0db270f74b"
        self.img_path = "/home/hh/realman_robotic_arm/src/rm_robot_control/pictures/inputs"
        self.output_dir = "/home/hh/realman_robotic_arm/src/rm_robot_control/pictures/outputs"
        # 初始化配置、客户端
        self.config = Config(self.api_token)
        self.client = Client(self.config)
        # 边界框、中心点、掩码、置信度、标签、类别名称、类别ID
        self.boxes, self.centers, self.masks, self.confidences, self.labels, self.class_names, self.class_ids = [], [], [], [], [], [], []
        # 所有物体信息
        self.all_object_info = []
    
    def recognition_and_decoding(self, image_b64, depth_frame, depth_intri):
        "-------------------------------- 运行 V2 模型 --------------------------------"
        api_path="/v2/task/dinox/detection"
        api_body={
            "model": "DINO-X-1.0",
            "image": image_b64,
            "prompt": {
                "type": "universal",
            },
            "targets": ["bbox", "mask"],
            "bbox_threshold": 0.25,
            "iou_threshold": 0.8
        }
        task = V2Task(api_path=api_path, api_body=api_body)
        self.client.run_task(task)
        result = task.result
        objects = result["objects"]
        "-------------------------------- 解码预测结果 --------------------------------"
        # 初始化临时变量
        class_counts = defaultdict(int)  # 用于跟踪每个类别的计数
        count = 0                        # 物体计数器
        CONF_THRESHOLD = 0.6             # 置信度阈值
        # 获取类别名称和 ID
        classes = [obj["category"] for obj in objects]
        classes = list(set(classes))
        class_name_to_id = {name: id for id, name in enumerate(classes)}
        # 将 RLE 编码的内容转换为数组
        for idx, obj in enumerate(objects):
            score = obj["score"]
            if score < CONF_THRESHOLD:
                continue
            self.boxes.append(obj["bbox"])
            self.masks.append(rle_to_array(obj["mask"]["counts"], obj["mask"]["size"][0] * obj["mask"]["size"][1]).reshape(obj["mask"]["size"]))
            self.confidences.append(obj["score"])
            cls_name = obj["category"].lower().strip()
            self.class_names.append(cls_name)
            self.class_ids.append(class_name_to_id[cls_name])
        self.boxes = np.array(self.boxes)
        self.masks = np.array(self.masks)
        self.class_ids = np.array(self.class_ids)
        # 为每个类别创建标签，并跟踪计数
        for class_name, confidence in zip(self.class_names, self.confidences):
            class_counts[class_name] += 1
            idx = class_counts[class_name]
            self.labels.append(f"{class_name}{idx}")
        # 输出识别结果
        print(f"检测到 {len(self.boxes)} 个物体")
        for i, (box, mask) in enumerate(zip(self.boxes, self.masks)):
            # 计算物体的六维位姿
            theta, center = self.compute_angle_with_mask(mask, depth_frame, depth_intri)
            pose_xyz, rot_mat = self.estimate_pose_from_mask(mask, depth_frame, depth_intri)
            if pose_xyz is not None:
                euler = R.from_matrix(rot_mat).as_euler('xyz', degrees=True)
                self.all_object_info.append({
                    "label": self.labels[i],
                    "XYZ": pose_xyz.tolist(),
                    "rotation_matrix": rot_mat.tolist(),
                    "euler_angles": euler.tolist(),
                    "theta": theta,
                    "center": center
                })
                print(f"物体 {count + 1}: 标签 = {self.labels[i]}, 置信度 = {self.confidences[i]:.3f}, 六维位姿 = ({pose_xyz[0]:.3f}, {pose_xyz[1]:.3f}, {pose_xyz[2]:.3f}, {euler[0]:.3f}, {euler[1]:.3f}, {euler[2]:.3f}), "
                      f"朝向角 = ({theta:.3f})°, 中心点 = ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
            else:
                print(f"物体 {count + 1}: 标签 = {self.labels[i]}: 无法估计姿态")
            count += 1

        return self.all_object_info
    
    # 可视化检测结果
    def visualization(self, color_image):
        # 创建 Supervision 检测对象
        detections = sv.Detections(
            xyxy = self.boxes,
            mask = self.masks.astype(bool),
            class_id = self.class_ids,
        )
        # 可视化检测结果
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=color_image.copy(), detections=detections)
        # 添加标签注释
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=self.labels)
        cv2.imwrite(os.path.join(self.output_dir, "annotated_demo_image.jpg"), annotated_frame)
        # 添加掩码注释
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(self.output_dir, "annotated_demo_image_with_mask.jpg"), annotated_frame)

        print(f"Annotated image has already been saved to {self.output_dir}")

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

if __name__ == "__main__":

    # 初始化 DINO-X 对象检测器
    DINO = DINOXObjectDetector()

    # 初始化 RealSense
    desired_serial = '327522301155'
    realsense = realsense(desired_serial)
    # 获取一帧图像
    for _ in range(10): realsense.pipeline.wait_for_frames()
    print("准备从 Realsense 获取一帧图像... 按 's' 键保存当前帧并进行处理")
    while True:
        depth_intri, depth_frame, color_image, depth_image = realsense.get_aligned_images()
        cv2.imshow("RGB Frame (Press 's' to select)", color_image)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(DINO.img_path, f"realsense_frame_{timestamp}.jpg")
        cv2.imwrite(save_path, color_image)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.destroyAllWindows()
            break
    image_path = os.path.join(DINO.img_path, f"realsense_frame_{timestamp}.jpg")
    image_b64 = image_to_base64(image_path)

    DINO.recognition_and_decoding(image_b64, depth_frame, depth_intri)
    DINO.visualization(color_image)