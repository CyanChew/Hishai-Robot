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

# === 检查状态合法性 ===
def is_valid_qpos(qpos: np.ndarray) -> bool:
    return np.isfinite(qpos).all() and (np.abs(qpos) < 1e3).all()

def wait_for_valid_robot_state(env: BaseEnv, max_attempts: int = 50):
    for i in range(max_attempts):
        qpos = env.agent.robot.get_qpos()
        if is_valid_qpos(qpos):
            print(f"✅ 第 {i} 次读取成功，机器人初始状态合法")
            return
        print(f"⚠️ 第 {i} 次读取 qpos 非法，等待中...")
        time.sleep(0.05)
    raise RuntimeError("❌ 多次尝试后仍未获得合法的机器人状态")

# === JoyCon + JoyLo 到 ManiSkill 的桥接 ===
class JoyToManiSkillBridge:
    def __init__(self, joylo: JoyLoController):
        self.joylo = joylo
        self.alpha = 0.95
        self.left_q = None
        self.right_q = None
        self.curr_torso_q = np.zeros(4)

    def get_action(self):
        joylo_action = self.joylo.act(curr_torso_q=self.curr_torso_q)
        self.curr_torso_q = joylo_action["torso_cmd"]

        self.left_q = joylo_action["arm_cmd"]["left"] if self.left_q is None else (1 - self.alpha) * self.left_q + self.alpha * joylo_action["arm_cmd"]["left"]
        self.right_q = joylo_action["arm_cmd"]["right"] if self.right_q is None else (1 - self.alpha) * self.right_q + self.alpha * joylo_action["arm_cmd"]["right"]

        gripper_cmd = joylo_action["gripper_cmd"]
        left_grip = gripper_cmd["left"]
        right_grip = gripper_cmd["right"]

        return np.concatenate([
            joylo_action["mobile_base_cmd"],
            self.curr_torso_q,
            self.left_q,
            self.right_q,
            np.array([left_grip, right_grip])
        ])[None, :]

# === 参数解析 ===
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--robot-uid", type=str, default="my_R1")
    parser.add_argument("-c", "--control-mode", type=str, default="pd_joint_pos")
    parser.add_argument("--sim-freq", type=int, default=100)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("-s", "--seed", type=int, default=None)
    return parser.parse_args(args)

# === 主程序 ===
def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    env = gym.make(
        "EmptyGroundR1-v0",
        obs_mode="none",
        reward_mode="none",
        render_mode="human",
        control_mode=args.control_mode,
        robot_uids=args.robot_uid,
        sim_config={"sim_freq": args.sim_freq, "control_freq": args.control_freq},
    )
    
    qpos = env.unwrapped.agent.robot.get_qpos()
    if hasattr(qpos, "detach"):  # torch tensor
        qpos = qpos.detach().cpu().numpy()

    if qpos.ndim == 2:
        qpos = qpos[0]  # 去除 batch 维度 (1, 23) → (23,)

    print("修正后的 qpos 长度：", len(qpos))
    assert len(qpos) >= 19, f"qpos 长度不足：{len(qpos)}，不能设置左右臂姿态"

    qpos[7:13] = np.array([1.56, 2.94, -2.54, 0, 0, 0])       # 左臂
    qpos[13:19] = np.array([-1.56, 2.94, -2.54, 0, 0, 0])     # 右臂
    env.agent.robot.set_qpos(qpos)
    env.agent.controller.reset()
    env.reset(seed=args.seed or 0)
    env: BaseEnv = env.unwrapped
    wait_for_valid_robot_state(env)

    viewer = env.render()
    viewer.paused = True

    print("🕹️ 启动 JoyCon + JoyLo 控制器（仿真环境）")

    # 关闭 ROS，直接用 JoyCon 本地控制
    joycon = R1JoyConInterface(ros_publish_functional_buttons=False, init_ros_node=False, gripper_toggle_mode=True)
    print("启动 JoyCon + JoyLo 控制器（仿真环境）")
    joylo_arms = JoyLoArmPositionController(
        left_motor_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        right_motor_ids=[8, 9, 10, 11, 12, 13, 14, 15],
        motors_port="/dev/tty_joylo",
        left_arm_joint_signs=[-1, 1, 1, 1, -1, 1],
        right_arm_joint_signs=[-1, 1, -1, 1, -1, 1],
        left_slave_motor_ids=[1, 3], left_master_motor_ids=[0, 2],
        right_slave_motor_ids=[9, 11], right_master_motor_ids=[8, 10],
        left_arm_joint_reset_positions=np.array([1.56, 2.94, -2.54, 0, 0, 0]),
        right_arm_joint_reset_positions=np.array([-1.56, 2.94, -2.54, 0, 0, 0]),
        multithread_read_joints=True,
    )
    print("启动 JoyCon成功")
    joylo = JoyLoController(joycon=joycon, joylo_arms=joylo_arms)
    print("启动 JoyCon成功2")
    bridge = JoyToManiSkillBridge(joylo)
    qpos = env.agent.robot.get_qpos()
    print("🤖 初始 qpos =", qpos)
    print("🔢 qpos shape =", qpos.shape)
    # === 控制循环 ===
    try:
        while True:
            action = bridge.get_action()
            env.step(action)
            viewer = env.render()
    except KeyboardInterrupt:
        print("\n🛑 收到 Ctrl+C，退出中...")

    joylo_arms.close()
    env.close()

if __name__ == "__main__":
    main()
