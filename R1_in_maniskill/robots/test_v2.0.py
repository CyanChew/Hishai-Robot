
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test-R1 with JoyCon+JoyLo in EmptyGround (No ROS, Auto Start)
"""

import argparse
import time
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

from mani_skill.agents.controllers.base_controller import DictController
from mani_skill.envs import BaseEnv

import sys
sys.path.append("/home/pine/R1_maniskill/R1_in_maniskill/envs/")
sys.path.append("/home/pine/R1_maniskill/R1_in_maniskill/robots/")

import iot_kitchen
import my_R1
import empty_ground_env
from brs_ctrl.joylo import JoyLoController
from brs_ctrl.joylo.joylo_arms import JoyLoArmPositionController
from brs_ctrl.joylo.joycon import R1JoyConInterface
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--robot-uid", type=str, default="my_R1")
    parser.add_argument("-c", "--control-mode", type=str, default="pd_joint_pos")
    parser.add_argument("--sim-freq", type=int, default=100)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("-s", "--seed", type=int, default=None)
    return parser.parse_args(args)
args = parse_args()
env = gym.make(
        "EmptyGroundR1-v0",
        obs_mode="state_dict",
        reward_mode="none",
        render_mode="human",
        control_mode="manual_joint_position",
        robot_uids=args.robot_uid,
        sim_config={"sim_freq": args.sim_freq, "control_freq": args.control_freq},
    )
# 配置 JoyLo 机械臂控制器
joycon = R1JoyConInterface(ros_publish_functional_buttons=False, init_ros_node=False, gripper_toggle_mode=True)
print("启动 JoyCon + JoyLo 控制器（仿真环境）")
joylo_arms = JoyLoArmPositionController(
    left_motor_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    right_motor_ids=[8, 9, 10, 11, 12, 13, 14, 15],
    motors_port="/dev/tty_joylo",
    left_arm_joint_signs=[-1, 1, 1, 1, -1, 1],
    right_arm_joint_signs=[-1, 1, -1, 1, -1, 1],
    left_slave_motor_ids=[1, 3],
    left_master_motor_ids=[0, 2],
    right_slave_motor_ids=[9, 11],
    right_master_motor_ids=[8, 10],
    left_arm_joint_reset_positions=np.array([1.56, 2.94, -2.54, 0, 0, 0]),
    right_arm_joint_reset_positions=np.array([-1.56, 2.94, -2.54, 0, 0, 0]),
    multithread_read_joints=True,
)

# 初始化 JoyCon + 控制组合器

joylo = JoyLoController(joycon=joycon, joylo_arms=joylo_arms)

# 初始化 ManiSkill 环境，必须支持 manual_joint_position
#env = gym.make("EmptyGroundR1-v0", control_mode="manual_joint_position", obs_mode="state")

obs, _= env.reset()
print(obs.keys())
def convert_to_maniskill_action(joylo_action):
    """ 把 JoyLo 控制器返回的 action 转换为 ManiSkill 可识别的动作 """
    return {
        "left_arm": joylo_action["arm_cmd"]["left"],
        "right_arm": joylo_action["arm_cmd"]["right"],
        "torso": joylo_action["torso_cmd"],
        "gripper": {
            "left": joylo_action["gripper_cmd"]["left"],
            "right": joylo_action["gripper_cmd"]["right"]
        }
    }

# 主循环：按照 JoyLoController 控制手指行为
from pprint import pprint
pprint(obs["agent"])
while True:
    # 读取 JoyCon + 机械臂状态，返回 action dict
    joylo_action = joylo.act(curr_torso_q=obs["agent"]["qpos"][0][3:7].numpy())

    # 转化成 ManiSkill 要求的动作
    action = convert_to_maniskill_action(joylo_action)

    # 步进一步
    obs, reward, done, truncated, info = env.step(action)
    env.render()

    if done or truncated:
        obs = env.reset()
