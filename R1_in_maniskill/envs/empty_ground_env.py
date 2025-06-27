# 文件位置建议：R1_in_maniskill/envs/empty_ground_env.py

import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env


@register_env("EmptyGroundR1-v0", max_episode_steps=1000)
class EmptyGroundR1Env(BaseEnv):
    SUPPORTED_ROBOTS = ["my_R1"]
    CONTROL_MODES = ["manual_joint_position"]

    def __init__(self, *args, robot_uids="my_R1", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        # 出生高度略高于地面，避免初始穿插
        spawn_pose = sapien.Pose([0, 0, 0.02], [1, 0, 0, 0])
        super()._load_agent(options, spawn_pose)

    def _load_scene(self, options: dict):
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light(direction=[1, -1, -1], color=[1, 1, 1])

        # 地面：4m × 4m × 0.01m
        floor_half = [2, 2, 0.005]
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=floor_half)
        builder.add_box_visual(
            half_size=floor_half,
            material=sapien.render.RenderMaterial(base_color=[0.8, 0.8, 0.8, 1]),
        )
        builder.initial_pose = sapien.Pose([0, 0, -floor_half[2]])
        builder.build_static(name="ground")

    def _initialize_episode(self, env_idx, options):
        # 可选：重置机器人状态
        pass

    def step(self, action):
        assert self.control_mode == "manual_joint_position"
        self.agent.set_qpos_from_action(action)  # 自定义关节控制接口
        #self.scene.step()
       # self._update_obs()
        return self.get_obs(), self.evaluate(), False, False, {}

    def evaluate(self):
        return {"success": self.agent.robot.get_qpos()[0] < 999}  # 永远失败，方便调试
