import numpy as np
# from gym.envs.robotics import rotations, robot_env, utils
import gym
import os
import math
import pybullet as p
from gym_myrobot.envs.robot_envfor_slide import RobotEnvSlide
from gym.utils import seeding
import pybullet_data
import random
import  time
from gym import spaces

'''
机械臂的reach到随机点的环境
输入动作为 arm的四个关节的delta量，范围在[-0.005,0.005]之间， 是个4 维数组
输出状态为字典形式的 observation{obs, achieved_goal, desired_goal}
                  其中，obs是目前的机械臂末端的 xyz以及euler的三元朝向，是个6维数组
升级版，可以在达到目标后锁死爪子
'''

def goal_distance(goal_a, goal_b):
    '''计算到目标位置的距离的差'''
    assert goal_a.shape == goal_b.shape
    # 计算两个坐标的差的2范数
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class SlideEnv(gym.GoalEnv):
    '''
    继承gymEnv
    '''
    def __init__(self,
                 n_substeps=10,                 # 每次给的0.005的deltajoint的值，需要大概40/200=0.2s的时间才能执行完毕
                 distance_threshold=0.05,      # 设置终点和目标点之间距离的限制成功标准，因为方块比较小，经过实验，0.005是一个很好的限制范围，达到这个范围内可以很精确的重合 
                 reward_type='sparse', 
                 usegui=False,
                 target_range=0.15,
                 ):
        IS_USEGUI = usegui
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.n_substeps=n_substeps
        self.n_actions=4
        self.targetUid=-1   # 每次reset都会把上次产生的目标点位删除，防止产生一堆点
        self._urdfRoot = pybullet_data.getDataPath()
        self.seed()
        # 是否进行渲染: 默认是false，但是可以创建环境的时候更改一下
        if IS_USEGUI:
            self.phisics = p.connect(p.GUI)
        else:
            self.phisics = p.connect(p.DIRECT)
        
        # 加载机器人，需要用到基环境
        self._robot = RobotEnvSlide()
        self._timeStep = 1. / 200.
        action_dim = 4
        self._action_bound = 0.05   
        action_high = np.array([self._action_bound] * action_dim)
        self.rest_poses = [0.04601942375302315, 0.0, 0.4924078583717346, 1.0906603336334229]
        # self.rest_poses = [0.000, 0.000 ,0.000 ,0.000 ,0.010, 0.010]
        self.target_range = target_range
        # 给出所有的要设置的关节位置需要得到初始的 xpos
        for i in range(4):
            p.resetJointState(self._robot.pandaUid, i, self.rest_poses[i], 0)  # 设置初始化的位置，以及速度
        p.setTimeStep(self._timeStep)
        end_pos_orn = np.array(self._robot.getObservation())
        self.initial_gripper_pos = np.array(end_pos_orn[:3]).reshape(-1)

        # 重置环境
        self.reset()
        obs = self._get_obs()

        # 空间设置
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                ),
            )
        )
        

    def compute_reward(self, achieved_goal, goal, info):
        '''计算奖励，有不稀疏和稀疏两种计算方法'''
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d
    
    def step(self, action):
        '''直接赋给目标关节值并渲染'''
        action = np.clip(action, -self._action_bound, self._action_bound)
        # 把action加上现在的joints，传入仿真并执行
        action_excute = action + self.joint_values_now
        self._set_action(action_excute)
        # 不再从仿真器中获取joints，而是直接把家和之后的结果赋给现在的joints
        self.joint_values_now = action_excute
        
        for _ in range(self.n_substeps):
            '''仿真n个pybullet时间步数，因为怕执行不完'''
            p.stepSimulation()
        
        obs = self._get_obs()
        done = False
        info = {
            'is_success':self._is_success(obs['achieved_goal'], self.goal),
        }

        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    
    def reset(self):
        '''重置所有位置，包括机械臂以及目标位置，目标位置用一个没有实体的红色小方块来表示， 这个urdf可以加载进来我们机械臂末端小方块的模型，但是不实体化,
           在这里，目标位置的设置是随机点，需要大改！这里需要好好改一下才能用。
        '''
        for i in range(4):
            p.resetJointState(self._robot.pandaUid, i, self.rest_poses[i], 0)  # 设置初始化的位置，以及速度
        p.setTimeStep(self._timeStep)

        # 记录初始时刻的前四个关节的值用于直接迭代循环关节值
        self.joint_values_now = np.array(self.rest_poses[:4])
        
        # Cube Pos
        for _ in range(100):
            xpos = 0.1 +0.15 * random.random()  # 0.25
            ypos = (random.random() * 0.2) - 0.1  # 0.10 0.50
            zpos = 0.02
            ang = 3.14 * 0.5 + 3.1415925438 * random.random()
            orn = p.getQuaternionFromEuler([0, 0, ang])
            # target Position：
            xpos_target = 0.2 +0.15 * random.random()  # 0.25
            ypos_target = (random.random() * 0.2) + 0.1  # 0.10 0.50
            zpos_target = 0.02
            ang_target = 3.14 * 0.5 + 3.1415925438 * random.random()
            orn_target = p.getQuaternionFromEuler([0, 0, ang_target])
            self.dis_between_target_block = math.sqrt(
                (xpos - xpos_target) ** 2 + (ypos - ypos_target) ** 2 + (zpos - zpos_target) ** 2)
            if self.dis_between_target_block >= 0.15:
                break
        
        if self.targetUid == -1:  
            self.targetUid = p.loadURDF("./gym_myrobot/envs/cube_small_target_slide.urdf",    # 根据这个urdf来改我们之前的末端小方块的urdf
                                        [xpos_target, ypos_target, zpos_target],
                                         orn_target, useFixedBase=1)                         # 目标方块位置固定，如果不使用useFixedBase=1，目标方块会坠落！
            self.blockUid = p.loadURDF("./gym_myrobot/envs/cube_small_slide.urdf",xpos, ypos, zpos,
                                       orn[0], orn[1], orn[2], orn[3])
        else:
            p.removeBody(self.targetUid)
            self.targetUid = p.loadURDF("./gym_myrobot/envs/cube_small_target_slide.urdf",    # 根据这个urdf来改我们之前的末端小方块的urdf
                                        [xpos_target, ypos_target, zpos_target],
                                         orn_target, useFixedBase=1)
            p.removeBody(self.blockUid)
            self.blockUid = p.loadURDF("./gym_myrobot/envs/cube_small_slide.urdf",xpos, ypos, zpos,
                                       orn[0], orn[1], orn[2], orn[3])

        p.setCollisionFilterPair(self.targetUid, self.blockUid, -1, -1, 0)
        self.goal=np.array([xpos_target,ypos_target,zpos_target])

        self._envStepCounter = 0
        obs = self._get_obs()
        self._observation = obs
        return self._observation


    def _set_action(self, action):
        '''利用基环境执行动作'''
        self._robot.applyAction(action)

    def _get_obs(self):
        '''获取机械臂的末端状态和朝向，规定字典形式的observation，由obs 和achieved goal与 desired goal组成'''
        end_pos_orn = np.array(self._robot.getObservation())
        # 前三个是xyz坐标
        end_pos = end_pos_orn[:3]
        # 后四个是orn四元量
        # 将得到的四元量的orn转换成Euler的叁元量，这个转换的函数对array生效，所以可以在真实环境中也使用！
        end_orn_quaternion = end_pos_orn[3:]
        end_orn = p.getEulerFromQuaternion(end_orn_quaternion)
        end_orn = np.array(end_orn)
        # 物体位置、姿态
        blockPos, blockOrn_temp = p.getBasePositionAndOrientation(self.blockUid)
        blockPos = np.array(blockPos)
        blockOrn = p.getEulerFromQuaternion(blockOrn_temp)
        blockOrn = np.array(blockOrn)
        # 目标位置，作为desiredgoal
        target_pos = np.array(p.getBasePositionAndOrientation(self.targetUid)[0])
        # obs 是两个array组成的列表，长度为0，1
        obs = [
            end_pos.flatten(),
            end_orn.flatten(),
            blockPos.flatten(),
            blockOrn.flatten(),
        ]

        achieved_goal = np.array(p.getBasePositionAndOrientation(self.blockUid)[0])

        # 把两个array组成的列表展开成一行array
        for i in range(1, len(obs)):
            end_pos = np.append(end_pos, obs[i])
        obs = end_pos.reshape(-1)

        self._observation = obs

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': target_pos.flatten(),
        }

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _is_success(self, achieved_goal, desired_goal):
        '''根据末端位置和目标末端位置的距离判断是否成功'''
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

