#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IoTKitchen-v0 环境：从 PartNet-Mobility 下载并加载三层抽屉模型示例
- 地面：4.2m × 4.7m × 0.02m
- 四面围墙：0.8m 高，0.05m 厚
- 北墙橱柜台面、灶台、中央餐桌（静态几何）
- 动态加载 PartNet-Mobility 模型 (Numeric ID=44962，三层抽屉)
- 注册自定义机器人 my_R1
"""

import os
import sapien.core as sapien
import torch
import numpy as np

# 导入 SAPIEN 下载 PartNet-Mobility 接口
import sapien

# —— 兼容不同版本的 ManiSkill BaseEnv、register_env —— #
try:
    from mani_skill3.envs.sapien_env import BaseEnv
    from mani_skill3.utils.registration import register_env
except ImportError:
    try:
        from mani_skill.envs.sapien_env import BaseEnv
        from mani_skill.utils.registration import register_env
    except ImportError:
        from mani_skill.envs.build_env import SapienEnv as BaseEnv
        from mani_skill.utils.registration import register_env
# —— 兼容导入结束 —— #


@register_env("IoTKitchen-v0", max_episode_steps=1000)
class IoTKitchenEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["my_R1"]

    def __init__(self, *args, robot_uids="my_R1", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        # 机器人出生在高位置
        spawn_pose = sapien.Pose([10, 0, 1.0], [1, 0, 0, 0])  # 1米高
        super()._load_agent(options, spawn_pose)

        
    def _load_scene(self, options: dict):
        # 添加环境光照
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        
        # 添加方向光
        light = self.scene.add_directional_light(
            direction=[1, -1, -1],
            color=[1, 1, 1]
        )
        
        quat_I = [1, 0, 0, 0]

        # ——— (1) 静态几何 ——— #
        # 地面：4.2m × 4.7m × 0.02m
        floor_half = [4.2 / 2, 4.7 / 2, 0.0001]
        b_floor = self.scene.create_actor_builder()
        b_floor.add_box_collision(half_size=floor_half)
        b_floor.add_box_visual(
            half_size=floor_half,
            material=sapien.render.RenderMaterial(base_color=[0.8, 0.8, 0.8, 1]),
        )
        b_floor.initial_pose = sapien.Pose([0, 0, -floor_half[2]], quat_I)
        b_floor.build_static(name="floor")
        
        # 中央餐桌：1.4m × 1.0m × 0.03m，高度0.8m
        # 餐桌桌面
        table_half = [0.7, 0.5, 0.015]

        # 餐桌四条腿
        leg_half = [0.03, 0.03, 0.4]  # 桌腿尺寸：6cm x 6cm x 80cm
        leg_positions = [
            [0.65, -0.8, 0.4],   # 右前
            [0.65, 0, 0.4],    # 右后
            [-0.65, -0.8, 0.4],  # 左前
            [-0.65, 0, 0.4]    # 左后
        ]

        # 先创建桌腿
        for i, pos in enumerate(leg_positions):
            b_leg = self.scene.create_actor_builder()
            b_leg.add_box_collision(half_size=leg_half)
            b_leg.add_box_visual(
                half_size=leg_half,
                material=sapien.render.RenderMaterial(base_color=[0.55, 0.35, 0.2, 1]),
            )
            b_leg.initial_pose = sapien.Pose(pos, quat_I)
            b_leg.build_static(name=f"table_leg_{i}")

        # 再创建桌面
        b_table = self.scene.create_actor_builder()
        b_table.add_box_collision(half_size=table_half)
        b_table.add_box_visual(
            half_size=table_half,
            material=sapien.render.RenderMaterial(base_color=[0.55, 0.35, 0.2, 1]),
        )
        b_table.initial_pose = sapien.Pose([0, -0.4, 0.8 - table_half[2]], quat_I)
        b_table.build_static(name="dining_table")
            # 中央餐桌：1.6m × 1.6m × 0.08m，高度0.8m


        # 围墙：高度 0.8m，厚度 0.05m
        wall_h, wall_t = 3, 0.02

        def _build_wall(half_size, pos, name):
            wb = self.scene.create_actor_builder()
            wb.add_box_collision(half_size=half_size)
            wb.add_box_visual(
                half_size=half_size,
                material=sapien.render.RenderMaterial(base_color=[0.9, 0.9, 0.9, 1]),
            )
            wb.initial_pose = sapien.Pose(pos, quat_I)
            wb.build_static(name=name)

        # X 方向两墙
        for sx in (-1, 1):
            half = [wall_t, 4.7 / 2, wall_h / 2]
            pos = [sx * (4.2 / 2 + wall_t / 2), 0, wall_h / 2]
            name = f"wall_x_{'pos' if sx > 0 else 'neg'}"
            _build_wall(half, pos, name)

        # Y 方向两墙
        for sy in (-1, 1):
            half = [4.2 / 2, wall_t, wall_h / 2]
            pos = [0, sy * (4.7 / 2 + wall_t / 2), wall_h / 2]
            name = f"wall_y_{'pos' if sy > 0 else 'neg'}"
            _build_wall(half, pos, name)

        # 北墙台面：3.0m × 0.6m × 0.1m
        # 北墙台面：3.0m × 0.6m × 0.1m
        counter_half = [2.6 / 2, 0.635 / 2, 0.02]
        b_counter = self.scene.create_actor_builder()
        b_counter.add_box_collision(half_size=counter_half)
        b_counter.add_box_visual(
            half_size=counter_half,
            material=sapien.render.RenderMaterial(base_color=[0.90, 0.80, 0.65, 1]),
        )
        b_counter.initial_pose = sapien.Pose(
            [0, 4.7 / 2 - wall_t - counter_half[1], 0.8 + counter_half[2]],
            quat_I,
        )
        b_counter.build_static(name="countertop")

        # 台面下方的两层柜子（最左侧）
        # 台面下方的两层柜子（最左侧）
        cabinet_width = 0.55
        cabinet_depth = counter_half[1] * 2  # 与台面深度相同
        cabinet_height = 0.8  # 与台面高度相同
        
        # 柜体框架位置计算
        cabinet_x = -counter_half[0] + cabinet_width / 2  # 台面最左侧
        cabinet_y = 4.7 / 2 - wall_t - counter_half[1]  # 与台面Y位置相同
        cabinet_z = cabinet_height / 2  # 底部从地面开始
        
        # 底板
        bottom_thickness = 0.02
        bottom_half = [cabinet_width / 2, cabinet_depth / 2, bottom_thickness / 2]
        bottom_z = bottom_thickness / 2
        
        b_bottom = self.scene.create_actor_builder()
        b_bottom.add_box_collision(half_size=bottom_half)
        b_bottom.add_box_visual(
            half_size=bottom_half,
            material=sapien.render.RenderMaterial(base_color=[0.90, 0.80, 0.65, 1]),
        )
        b_bottom.initial_pose = sapien.Pose([cabinet_x, cabinet_y, bottom_z], quat_I)
        b_bottom.build_static(name="cabinet_bottom")
        
        # 中间隔板（分隔两层）
        shelf_thickness = 0.02
        shelf_half = [cabinet_width / 2, cabinet_depth / 2, shelf_thickness / 2]
        shelf_z = cabinet_height / 2  # 中间位置
        
        b_shelf = self.scene.create_actor_builder()
        b_shelf.add_box_collision(half_size=shelf_half)
        b_shelf.add_box_visual(
            half_size=shelf_half,
            material=sapien.render.RenderMaterial(base_color=[0.90, 0.80, 0.65, 1]),
        )
        b_shelf.initial_pose = sapien.Pose([cabinet_x, cabinet_y, shelf_z], quat_I)
        b_shelf.build_static(name="cabinet_shelf")
        
        # 柜子背板
        back_thickness = 0.02
        back_half = [cabinet_width / 2, back_thickness / 2, cabinet_height / 2]
        back_y = cabinet_y + cabinet_depth / 2 - back_thickness / 2
        
        b_back = self.scene.create_actor_builder()
        b_back.add_box_collision(half_size=back_half)
        b_back.add_box_visual(
            half_size=back_half,
            material=sapien.render.RenderMaterial(base_color=[0.90, 0.80, 0.65, 1]),
        )
        b_back.initial_pose = sapien.Pose([cabinet_x, back_y, cabinet_z], quat_I)
        b_back.build_static(name="cabinet_back")
        
        # 柜子侧板（左右两侧）
        side_thickness = 0.02
        side_half = [side_thickness / 2, cabinet_depth / 2, cabinet_height / 2]
        
        for sx in (-1, 1):
            side_x = cabinet_x + sx * (cabinet_width / 2 - side_thickness / 2)
            b_side = self.scene.create_actor_builder()
            b_side.add_box_collision(half_size=side_half)
            b_side.add_box_visual(
                half_size=side_half,
                material=sapien.render.RenderMaterial(base_color=[0.90, 0.80, 0.65, 1]),
            )
            b_side.initial_pose = sapien.Pose([side_x, cabinet_y, cabinet_z], quat_I)
            b_side.build_static(name=f"cabinet_side_{'left' if sx < 0 else 'right'}")
        # 北墙灶台：0.4m × 0.3m × 0.1m
        stove_half = [0.15, 0.15, 0.02]
        b_stove = self.scene.create_actor_builder()
        b_stove.add_box_collision(half_size=stove_half)
        b_stove.add_box_visual(
            half_size=stove_half,
            material=sapien.render.RenderMaterial(base_color=[0.85, 0.82, 0.75, 1]),
        )
        b_stove.initial_pose = sapien.Pose(
            [0.85, 4.7 / 2 - wall_t - counter_half[1], 0.8 + counter_half[2] + stove_half[2] * 2],
            quat_I,
        )
        b_stove.build_static(name="stove")

        # # 中央餐桌：1.6m × 1.6m × 0.08m
        # table_half = [0.8, 0.8, 0.04]
        # b_table = self.scene.create_actor_builder()
        # b_table.add_box_collision(half_size=table_half)
        # b_table.add_box_visual(
        #     half_size=table_half,
        #     material=sapien.render.RenderMaterial(base_color=[0.55, 0.35, 0.2, 1]),
        # )
        # b_table.initial_pose = sapien.Pose([0, -1.0, 0.7 + table_half[2]], quat_I)
        # b_table.build_static(name="dining_table")

        # ——— (2) 下载 & 加载 PartNet-Mobility 三层抽屉模型 ——— #

        # ——— (2) 下载 & 加载 PartNet-Mobility 三层抽屉模型 ——— #

        # (2.1) Token：请替换为你在当前网络环境下生成的有效 Token
        # ——— (2) 下载 & 加载 PartNet-Mobility 模型 ——— #

        # 模型配置：每个模型单独设置ID、缩放、位置、旋转
        # 模型配置：每个模型单独设置ID、缩放、位置、旋转
        model_configs = [
            {#cabinet (trible drawer)
                "id": 44962,
                "scale": 0.53,
                "position": [-1.0, 0.5, 0.1],
                "rotation": [1, 0, 0, 0]  # 四元数 [w, x, y, z]
            },
            {#cabinet (single drawer)
                "id": 45524,
                "scale": 0.53,
                "position": [1.0, -0.5, 0.15],
                "rotation": [0.7071, 0, 0, 0.7071]  # 绕Z轴旋转90度
            },
            {
                #cabinet (trible drawer)
                "id": 44962,  # 重复的模型ID
                "scale": 0.53,  # 不同的缩放
                "position": [0.5, 1.0, 0.12],  # 不同的位置
                "rotation": [0.5, 0.5, 0.5, 0.5]  # 不同的旋转
            },
            {
            # microwave
                "id": 7310,  # 重复的模型ID
                "scale": 0.32,  # 不同的缩放
                "position": [0.5, 1.0, 0.12],  # 不同的位置
                "rotation": [0.5, 0.5, 0.5, 0.5]  # 不同的旋转
            },
            {
            # tap
                "id": 811,  # 重复的模型ID
                "scale": 0.15,  # 不同的缩放
                "position": [0.5, 1.0, 0.4],  # 不同的位置
                "rotation": [0.5, 0.5, 0.5, 0.5]  # 不同的旋转
            },
            {
            # chair
                "id": 100532,  # 重复的模型ID
                "scale": 0.6,  # 不同的缩放
                "position": [0.5, 1.0, 0.4],  # 不同的位置
                "rotation": [0.5, 0.5, 0.5, 0.5]  # 不同的旋转
            },
                      {
            # chair
                "id": 100532,  # 重复的模型ID
                "scale": 0.6,  # 不同的缩放
                "position": [0.5, 1.0, 0.4],  # 不同的位置
                "rotation": [0.5, 0.5, 0.5, 0.5]  # 不同的旋转
            },          {
            # chair
                "id": 100532,  # 重复的模型ID
                "scale": 0.6,  # 不同的缩放
                "position": [0.5, 1.0, 0.4],  # 不同的位置
                "rotation": [0.5, 0.5, 0.5, 0.5]  # 不同的旋转
            },          {
            # chair
                "id": 100532,  # 重复的模型ID
                "scale": 0.6,  # 不同的缩放
                "position": [0.5, 1.0, 0.4],  # 不同的位置
                "rotation": [0.5, 0.5, 0.5, 0.5]  # 不同的旋转
            },
            {
            # pot
                "id": 100015,  # 重复的模型ID
                "scale": 0.18,  # 不同的缩放
                "position": [0.5, 1.0, 0.4],  # 不同的位置
                "rotation": [0.5, 0.5, 0.5, 0.5]  # 不同的旋转
            },
            
            
        ]

        # Token：请替换为你在当前网络环境下生成的有效 Token
        token = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJlbWFpbCI6Inpob3UwNDg0QGUubnR1LmVkdS5zZyIsImlwIjoiMTcyLjIwLjAuMSIs"
            "InByaXZpbGVnZSI6MSwiZmlsZU9ubHkiOnRydWUsImlhdCI6MTc0ODkyNzkwOSwiZXhw"
            "IjoxNzgwNDYzOTA5fQ."
            "GubVTdcVT1DMY3Pe3HDwa8ICiFxrc28f-Y2uYIrsh_s"
        )

        # 存储所有加载的articulations
        self.articulations = []

        # 循环加载每个模型
        for idx, config in enumerate(model_configs):
            model_id = config["id"]
            scale = config["scale"]
            position = config["position"]
            rotation = config["rotation"]
            
            print(f"▶ 正在下载 PartNet-Mobility 模型 ID={model_id} …")
            urdf_file = sapien.asset.download_partnet_mobility(model_id, token)
            print("✔ 下载完成，URDF 文件路径：", urdf_file)

            print(f"▶ 正在加载 URDF ID={model_id} (缩放={scale}, 位置={position}, 旋转={rotation}) …")
            
            # 为每个模型创建单独的URDF Loader
            urdf_loader = self.scene.create_urdf_loader()
            urdf_loader.fix_root_link = False
            urdf_loader.scale = scale  # 使用配置中的缩放比例
            # 确保每个articulation有唯一名称，即使模型ID相同
            urdf_loader.name = f"partnet_model_{model_id}_{idx}"
            
            art = urdf_loader.load(urdf_file)
            if art is None:
                raise RuntimeError(f"URDF 加载失败，检查模型 ID={model_id} 是否有效，或网格文件是否齐全。")

            # 使用配置中的位置和旋转设置模型姿态
            art.set_pose(sapien.Pose(position, rotation))
            
            # 保存所有articulations
            self.articulations.append(art)
            
            # 保存第一个作为主要的drawer_cabinet
            if idx == 0:
                self.drawer_cabinet = art

            print(f"✔ 模型 ID={model_id} 已加载完成:")
            print(f"  - 缩放比例: {scale}")
            print(f"  - 位置: {position}")
            print(f"  - 旋转: {rotation}")
            print(f"  - 名称: partnet_model_{model_id}_{idx}")
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # 每次重置时重新设置所有articulations的位置
        
        if hasattr(self, 'articulations'):
            # 重新应用模型配置
            model_configs = [
                {
                    "position": [1.1, 2, 0.4],
                    "rotation": [0.7071, 0, 0, 0.7071]
                },
                {
                    "position": [0, 2, 0.4],
                    "rotation": [0.7071, 0, 0, 0.7071]
                },
                {
                    "position": [-0.48, 2, 0.4],
                    "rotation": [0.7071, 0, 0, 0.7071]
                },
                {
                    "position": [-0.95, 2.2, 0.97],
                    "rotation": [0.7071, 0, 0, 0.7071]
                },
                {
                    "position": [0, 2.05, 0.9],
                    "rotation": [0.7071, 0, 0, 0.7071]
                },
                                {
                    "position": [0, 0.35, 0.48],
                    "rotation": [0.7071, 0, 0, 0.7071]
                },                {
                    "position": [0.8, -0.4, 0.48],
                    "rotation": [1, 0, 0, 0]
                },                {
                    "position": [-0.8, -0.4, 0.48],
                    "rotation": [0, 0, 0, 1]
                },                {
                    "position": [0, -1.1, 0.48],
                    "rotation": [-0.7071, 0, 0, 0.7071]
                },
                   {
                    "position": [0.85, 2 , 0.95],
                    "rotation": [-0.7071, 0, 0, 0.7071]
                },
                
            ]
            
            for idx, art in enumerate(self.articulations):
                if idx < len(model_configs):
                    config = model_configs[idx]
                    art.set_pose(sapien.Pose(config["position"], config["rotation"]))
    def evaluate(self) -> dict:
        # 没有成功条件，始终返回 False
        return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

