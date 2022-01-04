import os, inspect
from posixpath import join
from numpy.core.records import recarray
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data
import  time

class RobotEnv:
    '''
    升级版程序，可以在达到目标之后锁定爪子
    这是reach的基环境
    '''
    def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.maxVelocity = 4.8
        self.end_link = 6
        self.maxForce = 200.

        # 关节目标值的限定
        self.joints_limit_low = [-2.82743338823, -1.79070781255, -0.942477796077, -1.79070781255]
        self.joints_limit_high = [2.82743338823, 1.57079632679, 1.38230076758, 2.04203522483]

        # reset
        self.reset()

    def reset(self):
        '''初始化环境'''
        # 载入模型
        p.resetSimulation()
        urdfRootPath=pybullet_data.getDataPath()
        # 设置重力这块，需要仔细斟酌，因为如果要加入加持的物体，物体必须是有重力的，但实际机械臂有自锁功能，是不受重力影响的！
        p.setGravity(0, 0, -9.8) # 由于日后该环境将会加入其他物体，所以重力必须考虑！

        p.setTimeStep(self.timeStep)
        planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])
        # 使用相对路径
        self.pandaUid = p.loadURDF("./gym_myrobot/envs/open_manipulator_for_reach.urdf", useFixedBase=True)  # fixedbase 是吧机械臂的底座固定住，不然每次仿真都会乱跑！
        # 重设所有joint的位置
        rest_poses = [0.000, 0.000 ,0.000 ,0.000 ,0.010, 0.010]   # 前四个joint以及两个finger，要符合joint 的限定范围
        for i in range(6):
            p.resetJointState(self.pandaUid,i, rest_poses[i])

        # 放一个桌子
        self.tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.63])

    def getActionDimension(self):
        '''这个环境只控制前4个关节的delta值，所以维度为4'''
        return 4

    def getObservationDimension(self):
        '''得到obser的维度'''
        return len(self.getObservation)
    
    def getObservation(self):
        '''获取环境的obs, 是末端笛卡尔坐标和朝向四元量的连接'''
        observation = []
        state = p.getLinkState(self.pandaUid, self.end_link)
        pos = state[4]     # 4比0得到的位置信息更加精确！
        pos = list(pos)
        orn = state[5]
        orn = list(orn)
        observation.extend(pos)
        observation.extend(orn)
        
        return observation

    def applyAction(self, action):
        '''直接把joints值赋给机械臂'''
        jointPoses_target = action
        jointPoses_target = np.clip(jointPoses_target, self.joints_limit_low[:4], self.joints_limit_high[:4])
        p.setJointMotorControlArray(self.pandaUid, list(range(4)), p.POSITION_CONTROL, jointPoses_target)

    def operate_gripper(self, motor):
        '''单独控制爪子'''
        p.setJointMotorControlArray(self.pandaUid, [4,5], p.POSITION_CONTROL, [motor, motor])

    