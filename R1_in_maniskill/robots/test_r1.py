#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test-R1 in IoT-Kitchen
----------------------
在 IoTKitchen-v0 场景中加载自定义机器人 my_R1，并演示 500 帧随机/关键帧/静态渲染。

用法示例：
  python test_r1.py --robot-uid my_R1 --none-actions
  python test_r1.py --robot-uid my_R1 --random-actions
  python test_r1.py --robot-uid my_R1 --keyframe stand --keyframe-actions
"""

import argparse
import time

# ManiSkill-3 通常用 gymnasium；旧版 ManiSkill-0.x 用 gym
try:
    import gymnasium as gym
except ImportError:
    import gym

from mani_skill.agents.controllers.base_controller import DictController
import sys
sys.path.append("/home/pine/R1_maniskill/R1_in_maniskill/envs/")  # 确保能找到 mani_skill 包
sys.path.append("/home/pine/R1_maniskill/R1_in_maniskill/robots/")  # 确保能找到 robots 包
# —— 先导入以完成“注册”动作 —— #
import iot_kitchen
#import iot_kitchen
import my_R1  # 确保 my_R1 已经恰当用 @register_agent() 注册
import empty_ground_env
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--robot-uid", type=str, default="my_R1",
                        help="要放入环境的机器人 UID，需与注册时的 uid 相同。")
    parser.add_argument("-c", "--control-mode", type=str, default="pd_joint_pos",
                        help="控制模式，比如 'pd_joint_pos' 或 'pd_joint_delta_pos' 等。")
    parser.add_argument("-k", "--keyframe", type=str, default=None,
                        help="要使用的关键帧名称（可选）。")
    parser.add_argument("--keyframe-actions", action="store_true",
                        help="使用关键帧动作，让机器人保持 keyframe 姿态。")
    parser.add_argument("--random-actions", action="store_true",
                        help="每步随机采样动作并发送给机器人。")
    parser.add_argument("--none-actions", action="store_true",
                        help="不发送任何动作，仅更新渲染。")
    parser.add_argument("--zero-actions", action="store_true",
                        help="每步发送全零动作给机器人。")
    parser.add_argument("--sim-freq", type=int, default=100,
                        help="模拟频率 (Hz)，默认 100。")
    parser.add_argument("--control-freq", type=int, default=20,
                        help="控制频率 (Hz)，默认 20。")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="随机种子。若不指定，则不设种子。")
    return parser.parse_args(args)


def main():
    args = parse_args()

    # 如果指定了随机种子，则设置 numpy RNG（只影响随机 action）
    if args.seed is not None:
        import numpy as np
        np.random.seed(args.seed)

    # —— 创建 Gym 环境 —— #
    # env = gym.make(
    #     "IoTKitchen-v0",
    #     obs_mode="none",               # 环境不主动返回 obs，直接用 render 画面
    #     reward_mode="none",            # 不计算奖励
    #     enable_shadow=True,
    #     control_mode=args.control_mode,
    #     robot_uids=args.robot_uid,
    #     sensor_configs={"shader_pack": "default"},
    #     human_render_camera_configs={"shader_pack": "default"},
    #     viewer_camera_configs={"shader_pack": "default"},
    #     render_mode="human",           # “人类”渲染，将返回一个 Viewer
    #     sim_config={"sim_freq": args.sim_freq, "control_freq": args.control_freq},
    #     sim_backend="auto",
    # )
    #创建空地环境
    env = gym.make(
    "EmptyGroundR1-v0",
    obs_mode="none",
    reward_mode="none",
    render_mode="human",
    control_mode="pd_joint_pos",
    robot_uids="my_R1",
    )
    # reset 时会调用 IoTKitchenEnv._load_scene()、_load_agent()，
    # 先创建地面、墙、橱柜、灶台、餐桌，再实例化 my_R1
    env.reset(seed=args.seed or 0)

    # 告诉类型检查：env.unwrapped 是 BaseEnv
    env: BaseEnv = env.unwrapped  # type: ignore

    print(f"已加载机器人：{args.robot_uid}，控制模式：{args.control_mode}")
    print("可用关键帧：", list(env.agent.keyframes.keys()))
    qpos = env.agent.robot.get_qpos()
    print("🤖 初始 qpos =", qpos)
    print("🔢 qpos shape =", qpos.shape)

    # —— 如果有 keyframes，尝试初始化到某个关键帧 —— #
    kf = None
    if len(env.agent.keyframes) > 0:
        if args.keyframe is not None and args.keyframe in env.agent.keyframes:
            kf = env.agent.keyframes[args.keyframe]
            print(f"使用关键帧：{args.keyframe}")
        else:
            # 默认取第一个 keyframe
            kf_name, kf = next(iter(env.agent.keyframes.items()))
            print(f"未指定或无效关键帧，使用：{kf_name}")

        # 把机器人 qpos、qvel、pose 设置到关键帧
        if kf.qpos is not None:
            env.agent.robot.set_qpos(kf.qpos)
            env.agent.controller.reset()
        if kf.qvel is not None:
            env.agent.robot.set_qvel(kf.qvel)
        if kf.pose is not None:
            env.agent.robot.set_pose(kf.pose)

    # —— 如果开启了 GPU 模拟，需要同步一次 kinematics —— #
    if getattr(env, "gpu_sim_enabled", False):
        env.scene._gpu_apply_all()
        env.scene.px.gpu_update_articulation_kinematics()
        env.scene._gpu_fetch_all()

    # —— 获取 Viewer 并暂停渲染（让你先看静态效果） —— #
    viewer = env.render()
    viewer.paused = True
    viewer = env.render()


    # —— 交互主循环 —— #
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
            # 都没指定时仅更新渲染
            env.step(None)

        # 再次 render，更新画面
        viewer = env.render()


if __name__ == "__main__":
    main()
