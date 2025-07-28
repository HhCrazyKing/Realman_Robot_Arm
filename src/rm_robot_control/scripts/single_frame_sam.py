import os
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realsense import realsense
from rm_reference_code.SAM.segment_anything import sam_model_registry, SamPredictor

class SamPredict: 
    # 图片（any）、模型（any）、模型类型（str）、模型设备（str）、是否选择随机颜色覆盖（bool）
    def __init__(self, image, checkpoint, model_type, model_device, random_color: bool):
        self.image = image
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.model_device = model_device
        self.points = []
        self.labels = []
        self.marker_size = 300
        self.random_color = random_color

    # 人工标注前景和背景点
    def interactive_point_selection(self):
        
        fig, ax = plt.subplots()
        ax.imshow(self.image)

        def onclick(event):
            if event.xdata is None or event.ydata is None:
                return   # 点击范围外
            x, y = int(event.xdata), int(event.ydata)  
            if event.button == 1:  # 左键 -> 前景
                self.points.append([x, y])
                self.labels.append(1)
                ax.scatter(x, y, color='green', marker='*', s=self.marker_size, edgecolor='white')
            elif event.button == 3:   # 右键 -> 背景
                self.points.append([x, y])
                self.labels.append(0)
                ax.scatter(x, y, color='red', marker='*', s=self.marker_size, edgecolor='white')
            fig.canvas.draw()

        def onkey(event):
            if event.key == 'enter':
                plt.close()

        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        plt.axis('off')
        plt.show()

        print(f"标注完成，共有 {len(self.points)} 个点，前景 {sum(self.labels)}，背景 {len(self.labels) - sum(self.labels)}")
    
    # 使用SAM模型进行预测
    def get_result(self):
        
        # 设置模型
        model = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        model.to(device=self.model_device)
        predictor = SamPredictor(model)

        # 传入图片
        predictor.set_image(self.image)

        # 传入预标记点
        points = np.array(self.points)
        labels = np.array(self.labels)

        # 使用SAM_predictor返回覆盖、置信度及logits
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        print('原始图片高度Height为:', masks.shape[1])
        print('原始图片宽度Width为:', masks.shape[2])
        print('识别主体Mask次数为:', masks.shape[0])
        print('得分数组:', scores)

        # 选出得分最高的 mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]

        # 可视化结果
        plt.title(f"Best Mask | Score: {best_score:.3f}", fontsize=18)
        plt.axis('off')
        plt.imshow(self.image)

        # 展示标记点
        pos_points = points[labels == 1]
        neg_points = points[labels == 0]
        plt.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=self.marker_size, edgecolor='white', linewidth=1.25)
        plt.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=self.marker_size, edgecolor='white', linewidth=1.25)
        
        # 展示最佳mask
        if self.random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = best_mask.shape[-2:]
        mask_image = best_mask.reshape(h, w, 1) * color.reshape((1, 1, -1))
        plt.imshow(mask_image)
        plt.show()

        return best_mask

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
    selected_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # 输入模型
    sam_checkpoint = r'/home/hh/realman_robotic_arm/src/rm_robot_control/model/sam/sam_vit_l.pth'
    # 输入模型类型
    sam_model_type = 'vit_l'
    # 输入模型所需设备类型：'cuda'代表使用GPU
    sam_device = 'cuda'

    # 创建SamPredict类的实例
    model_one = SamPredict(image=selected_image, checkpoint=sam_checkpoint, model_type=sam_model_type, model_device=sam_device, random_color=False)
    # 启动鼠标交互式标注
    model_one.interactive_point_selection()
    # 获取预测结果
    model_one.get_result()