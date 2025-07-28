import os
import sys
import cv2
import numpy as np
import supervision as sv

from datetime import datetime
from collections import defaultdict

from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.image_resizer import image_to_base64
from dds_cloudapi_sdk.tasks.v2_task import V2Task

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realsense import realsense
from rm_reference_code.DINO_X.rle_util import rle_to_array

# 超参数初始化
API_TOKEN = "0768f2df00d338f57200ac0db270f74b"
IMG_PATH = "/home/hh/realman_robotic_arm/src/rm_robot_control/pictures/inputs"
OUTPUT_DIR = "/home/hh/realman_robotic_arm/src/rm_robot_control/pictures/outputs"

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
    save_path = os.path.join(IMG_PATH, f"realsense_frame_{timestamp}.jpg")
    cv2.imwrite(save_path, color_image)
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.destroyAllWindows()
        break
image_path = os.path.join(IMG_PATH, f"realsense_frame_{timestamp}.jpg")
image = image_to_base64(image_path)
if os.path.exists(image_path):
    os.remove(image_path)

# 初始化配置、客户端
token = API_TOKEN
config = Config(token)
client = Client(config)

# 运行V2模型
api_path="/v2/task/dinox/detection"
api_body={
    "model": "DINO-X-1.0",
    "image": image,
    "prompt": {
        "type": "universal",
    },
    "targets": ["bbox", "mask"],
    "bbox_threshold": 0.25,
    "iou_threshold": 0.8
}

task = V2Task(
    api_path=api_path,
    api_body=api_body
)

client.run_task(task)
result = task.result

objects = result["objects"]

# 解码预测结果
classes = [obj["category"] for obj in objects]
classes = list(set(classes))
class_name_to_id = {name: id for id, name in enumerate(classes)}
# 初始化储存变量
class_counts = defaultdict(int)  # 用于跟踪每个类别的计数
count = 0                        # 物体计数器
CONF_THRESHOLD = 0.4             # 置信度阈值
boxes = []                       # 用于存储边界框
centers = []                     # 用于存储中心点
masks = []                       # 用于存储掩码
confidences = []                 # 用于存储置信度
labels = []                      # 用于存储标签
class_names = []                 # 用于存储类别名称
class_ids = []                   # 用于存储类别ID
# 将 RLE 编码的掩码转换为数组
for idx, obj in enumerate(objects):
    score = obj["score"]
    if score < CONF_THRESHOLD:
        continue
    boxes.append(obj["bbox"])
    masks.append(rle_to_array(obj["mask"]["counts"], obj["mask"]["size"][0] * obj["mask"]["size"][1]).reshape(obj["mask"]["size"]))
    confidences.append(obj["score"])
    cls_name = obj["category"].lower().strip()
    class_names.append(cls_name)
    class_ids.append(class_name_to_id[cls_name])
# 将列表转换为 NumPy 数组
boxes = np.array(boxes)
masks = np.array(masks)
class_ids = np.array(class_ids)

# 为每个类别创建标签
for class_name, confidence in zip(class_names, confidences):
    class_counts[class_name] += 1
    idx = class_counts[class_name]
    labels.append(f"{class_name}{idx}")

for i, (box,  cls_id) in enumerate(zip(boxes, class_ids)):
    # 计算中心点坐标
    x1, y1, x2, y2 = map(int, box)
    point = np.array([(x1 + x2) // 2, (y1 + y2) // 2])   # 中心点坐标
    label_numbered = labels[int(cls_id)]
    print(f"物体 {count + 1}: 标签 = {label_numbered}, 置信度 = {confidences[i]:.2f}, 中心点 = ({point[0]}, {point[1]})")
    count += 1

# 创建 Supervision 检测对象
detections = sv.Detections(
    xyxy = boxes,
    mask = masks.astype(bool),
    class_id = class_ids,
)
# 可视化检测结果
box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=color_image.copy(), detections=detections)
# 添加标签注释
label_annotator = sv.LabelAnnotator()
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
cv2.imwrite(os.path.join(OUTPUT_DIR, "annotated_demo_image.jpg"), annotated_frame)
# 添加掩码注释
mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
cv2.imwrite(os.path.join(OUTPUT_DIR, "annotated_demo_image_with_mask.jpg"), annotated_frame)

print(f"Annotated image has already been saved to {OUTPUT_DIR}")