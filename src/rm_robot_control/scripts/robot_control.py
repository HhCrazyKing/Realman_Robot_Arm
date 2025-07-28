import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from rm_reference_code.RMAPI_Python.Robotic_Arm.rm_robot_interface import *

class RobotArmController:
    def __init__(self, ip, port, level=3, mode=2):
        """
        机械臂的初始化与连接
        """
        self.thread_mode = rm_thread_mode_e(mode)
        self.robot = RoboticArm(self.thread_mode)
        self.handle = self.robot.rm_create_robot_arm(ip, port, level)

        if self.handle.id == -1:
            print("\nFailed to connect to the robot arm\n")
            exit(1)
        else:
            print(f"\nSuccessfully connected to the robot arm: {self.handle.id}\n")

    def disconnect(self):
        handle = self.robot.rm_delete_robot_arm()
        if handle == 0:
            print("\nSuccessfully disconnected from the robot arm\n")
        else:
            print("\nFailed to disconnect from the robot arm\n")

    def get_arm_software_info(self):
        """
        读取机械臂软件信息
        """
        software_info = self.robot.rm_get_arm_software_info()
        if software_info[0] == 0:
            print("\n================== Arm Software Information ==================")
            print("Arm Model: ", software_info[1]['product_version'])
            print("Algorithm Library Version: ", software_info[1]['algorithm_info']['version'])
            print("Control Layer Software Version: ", software_info[1]['ctrl_info']['version'])
            print("Dynamics Version: ", software_info[1]['dynamic_info']['model_version'])
            print("Planning Layer Software Version: ", software_info[1]['plan_info']['version'])
            print("==============================================================\n")
        else:
            print("\nFailed to get arm software information, Error code: ", software_info[0], "\n")

    def get_joint_degree(self):
        """
        获取当前关节角度
        """
        ret, joint_degree = self.robot.rm_get_joint_degree()

        return joint_degree

    def movej(self, joint, v=20, r=0, connect=0, block=1):
        """
        关节空间运动

        Args:
            joint (list): 各关节目标角度数组，单位：°
            v (int): 速度百分比系数，1~100
            r (int, optional): 交融半径百分比系数，0~100。
            connect (int): 轨迹连接标志
                - 0：立即规划并执行轨迹，不与后续轨迹连接。
                - 1：将当前轨迹与下一条轨迹一起规划，但不立即执行。阻塞模式下，即使发送成功也会立即返回。
            block (int): 阻塞设置
                - 多线程模式：
                    - 0：非阻塞模式，发送指令后立即返回。
                    - 1：阻塞模式，等待机械臂到达目标位置或规划失败后才返回。
                - 单线程模式：
                    - 0：非阻塞模式。
                    - 其他值：阻塞模式并设置超时时间，单位为秒。
        """
        movej_result = self.robot.rm_movej(joint, v, r, connect, block)
        if movej_result == 0:
            print("\nmovej motion succeeded\n")
        else:
            print("\nmovej motion failed, Error code: ", movej_result, "\n")

    def movej_p(self, pose, v=20, r=0, connect=0, block=1):
        """
        该函数用于关节空间运动到目标位姿

        Args:
            pose (list[float]): 目标位姿，位置单位：米，姿态单位：弧度。
            v (int): 速度百分比系数，1~100
            r (int, optional): 交融半径百分比系数，0~100。
            connect (int): 轨迹连接标志
                - 0：立即规划并执行轨迹，不与后续轨迹连接。
                - 1：将当前轨迹与下一条轨迹一起规划，但不立即执行。阻塞模式下，即使发送成功也会立即返回。
            block (int): 阻塞设置
                - 多线程模式：
                    - 0：非阻塞模式，发送指令后立即返回。
                    - 1：阻塞模式，等待机械臂到达目标位置或规划失败后才返回。
                - 单线程模式：
                    - 0：非阻塞模式。
                    - 其他值：阻塞模式并设置超时时间，单位为秒。
        """
        movej_p_result = self.robot.rm_movej_p(pose, v, r, connect, block)
        if movej_p_result == 0:
            print("\nmovej_p motion succeeded\n")
        else:
            print("\nmovej_p motion failed, Error code: ", movej_p_result, "\n")

    def movel(self, pose, v=20, r=0, connect=0, block=1):
        """
        笛卡尔空间直线运动

        Args:
            pose (list[float]): 目标位姿,位置单位：米，姿态单位：弧度
            v (int): 速度百分比系数，1~100
            r (int, optional): 交融半径百分比系数，0~100。
            connect (int): 轨迹连接标志
                - 0：立即规划并执行轨迹，不与后续轨迹连接。
                - 1：将当前轨迹与下一条轨迹一起规划，但不立即执行。阻塞模式下，即使发送成功也会立即返回。
            block (int): 阻塞设置
                - 多线程模式：
                    - 0：非阻塞模式，发送指令后立即返回。
                    - 1：阻塞模式，等待机械臂到达目标位置或规划失败后才返回。
                - 单线程模式：
                    - 0：非阻塞模式。
                    - 其他值：阻塞模式并设置超时时间，单位为秒。
        """
        movel_result = self.robot.rm_movel(pose, v, r, connect, block)
        if movel_result == 0:
            print("\nmovel motion succeeded\n")
        else:
            print("\nmovel motion failed, Error code: ", movel_result, "\n")

    def movec(self, pose_via, pose_to, v=20, r=0, loop=0, connect=0, block=1):
        """
        笛卡尔空间圆弧运动

        Args:
            pose_via (list[float]): 中间点位姿，位置单位：米，姿态单位：弧度
            pose_to (list[float]): 终点位姿，位置单位：米，姿态单位：弧度
            v (int): 速度百分比系数，1~100
            r (int, optional): 交融半径百分比系数，0~100。
            loop (int): 规划圈数.
            connect (int): 轨迹连接标志
                - 0：立即规划并执行轨迹，不与后续轨迹连接。
                - 1：将当前轨迹与下一条轨迹一起规划，但不立即执行。阻塞模式下，即使发送成功也会立即返回。
            block (int): 阻塞设置
                - 多线程模式：
                    - 0：非阻塞模式，发送指令后立即返回。
                    - 1：阻塞模式，等待机械臂到达目标位置或规划失败后才返回。
                - 单线程模式：
                    - 0：非阻塞模式。
                    - 其他值：阻塞模式并设置超时时间，单位为秒。
        """
        movec_result = self.robot.rm_movec(pose_via, pose_to, v, r, loop, connect, block)
        if movec_result == 0:
            print("\nmovec motion succeeded\n")
        else:
            print("\nmovec motion failed, Error code: ", movec_result, "\n")

    def forward_kinematics(self, joint, flag):
        """
        正解算法接口

        Args:
            joint (list[float]): 关节角度，单位：°
            flag (int, optional): 选择姿态表示方式，默认欧拉角表示姿态
                - 0: 返回使用四元数表示姿态的位姿列表[x,y,z,w,x,y,z]
                - 1: 返回使用欧拉角表示姿态的位姿列表[x,y,z,rx,ry,rz]
        """
        pose = self.robot.rm_algo_forward_kinematics(joint, flag)
        return pose

    def pos2matrix(self, pose):
        """
        位姿转旋转矩阵

        Args:
            pose (list[float]): 位置姿态列表[x,y,z,rx,ry,rz]

        """
        matrix = self.robot.rm_algo_pos2matrix(pose)
        matrix = np.array(matrix.data, dtype=np.float64).reshape(4, 4)

        return matrix
    
    def matrix2pos(self, matrix, flag):
        """
        旋转矩阵转位姿

        Args:
            matrix (rm_matrix_t): 旋转矩阵
            flag (int, optional): 选择姿态表示方式，默认欧拉角表示姿态
                - 0: 返回使用四元数表示姿态的位姿列表[x,y,z,w,x,y,z]
                - 1: 返回使用欧拉角表示姿态的位姿列表[x,y,z,rx,ry,rz]
                
        """
        flat_data = matrix.astype(np.float32).flatten()
        rm_mat = rm_matrix_t()
        rm_mat.data = (c_float * 16)(*flat_data)
        pose = self.robot.rm_algo_matrix2pos(rm_mat, flag)

        return pose

# 将旋转矩阵转换为齐次变换矩阵
def toHTM(rot):
    rot_HTM = np.eye(4)
    rot_HTM[:3, :3] = rot
    
    return rot_HTM

def main():
    # 创建一个机器人手臂控制器实例, 并连接到机器人手臂
    robot_controller = RobotArmController("192.168.1.18", 8080, 3)

    # 获取机械臂的基本信息
    robot_controller.get_arm_software_info()

    # 关节空间运动, 输入关节角度
    # joint =[0, -90, 0, 0, 0, -133.5880012512207] # 机械臂垂直向下
    # joint = [-4.533999919891357, -51.66699981689453, -48.915000915527344, 0.09700000286102295, 100.49600219726562, 184.46400451660156] # 机械臂末端的rpy为0
    # robot_controller.movej(joint)

    # 关节空间运动, 输入目标位姿
    # pos = [0.406, -0.032, 0.522, 0, 0, 0]
    pos = [0.42, 0.35, 0.1, np.pi/2, 0, np.pi/2]
    robot_controller.movej_p(pos)
    
    # 获取当前关节角度
    joint_degree = robot_controller.get_joint_degree()

    # 获取当前末端位姿
    pose = robot_controller.forward_kinematics(joint_degree, 1)
    print(pose)

    # 与机械臂断连
    robot_controller.disconnect()


if __name__ == "__main__":
    main()