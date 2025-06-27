#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test-R1 in IoT-Kitchen / EmptyGround
"""

import argparse
import time
import numpy as np
import math

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

# ===== 安全性检查函数 =====

def is_valid_qpos(qpos: np.ndarray) -> bool:
    """判断 qpos 是否合法（不含 nan 且数值合理）"""
    return np.isfinite(qpos).all() and (np.abs(qpos) < 1e3).all()

def wait_for_valid_robot_state(env: BaseEnv, max_attempts: int = 50):
    """在 reset 后等待机器人状态稳定（避免 qpos 为 nan）"""
    for i in range(max_attempts):
        qpos = env.agent.robot.get_qpos()
        if is_valid_qpos(qpos):
            print(f"✅ 第 {i} 次读取成功，机器人初始状态合法")
            return
        print(f"⚠️ 第 {i} 次读取 qpos 非法，等待中...")
        time.sleep(0.05)
    raise RuntimeError("❌ 多次尝试后仍未获得合法的机器人状态")

# ===== 参数解析 =====

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--robot-uid", type=str, default="my_R1")
    parser.add_argument("-c", "--control-mode", type=str, default="pd_joint_pos")
    parser.add_argument("-k", "--keyframe", type=str, default=None)
    parser.add_argument("--keyframe-actions", action="store_true")
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--none-actions", action="store_true")
    parser.add_argument("--zero-actions", action="store_true")
    parser.add_argument("--sim-freq", type=int, default=100)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("-s", "--seed", type=int, default=None)
    return parser.parse_args(args)


def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # ===== 创建环境 =====
    env = gym.make(
        "EmptyGroundR1-v0",
        obs_mode="none",
        reward_mode="none",
        render_mode="human",
        control_mode=args.control_mode,
        robot_uids=args.robot_uid,
        sim_config={"sim_freq": args.sim_freq, "control_freq": args.control_freq},
        sim_backend="auto",
    )
    env.reset(seed=args.seed or 0)

    env: BaseEnv = env.unwrapped

    # ===== 安全性检测（新增） =====
    wait_for_valid_robot_state(env)

    print(f"已加载机器人：{args.robot_uid}，控制模式：{args.control_mode}")
    print("可用关键帧：", list(env.agent.keyframes.keys()))

    # ===== 关键帧设置 =====
    kf = None
    if len(env.agent.keyframes) > 0:
        if args.keyframe is not None and args.keyframe in env.agent.keyframes:
            kf = env.agent.keyframes[args.keyframe]
            print(f"使用关键帧：{args.keyframe}")
        else:
            kf_name, kf = next(iter(env.agent.keyframes.items()))
            print(f"未指定或无效关键帧，使用：{kf_name}")

        if kf.qpos is not None:
            env.agent.robot.set_qpos(kf.qpos)
            env.agent.controller.reset()
        if kf.qvel is not None:
            env.agent.robot.set_qvel(kf.qvel)
        if kf.pose is not None:
            env.agent.robot.set_pose(kf.pose)

    if getattr(env, "gpu_sim_enabled", False):
        env.scene._gpu_apply_all()
        env.scene.px.gpu_update_articulation_kinematics()
        env.scene._gpu_fetch_all()

    viewer = env.render()
    viewer.paused = True
    viewer = env.render()

    # ===== 主循环 =====
    while True:
        if args.random_actions:
            act = env.action_space.sample()
            env.step(act)
        elif args.none_actions:
            env.step(None)
        elif args.zero_actions:
            zero_act = env.action_space.sample() * 0
            env.step(zero_act)
        elif args.keyframe_actions:
            assert kf is not None, "当前机器人没有可用关键帧，无法执行 keyframe-actions"
            if isinstance(env.agent.controller, DictController):
                env.step(env.agent.controller.from_qpos(kf.qpos))
            else:
                env.step(kf.qpos)
        else:
            env.step(None)

        viewer = env.render()


if __name__ == "__main__":
    main()
