import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
import torch
import time
import os
import math


def wrap_angle(angle):
    """归一化角度"""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


# === 余弦退火学习率函数 ===
def cosine_annealing_lr(initial_lr, total_timesteps):
    def lr_schedule(progress_remaining):
        t = 1 - progress_remaining  # 当前训练进度(0~1)
        lr = 0.5 * initial_lr * (1 + math.cos(math.pi * t))
        return lr
    return lr_schedule


class ViewerCallback(BaseCallback):
    def __init__(self, env, save_freq=10000, save_dir="./models", verbose=0):
        super().__init__(verbose)
        self.env = env
        self.viewer = mujoco.viewer.launch_passive(env.model, env.data)
        self.start_time = time.time()
        self.save_freq = save_freq
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        raw_angle = float(self.env.data.qpos[0])
        if self.env.clockwise_positive:
            angle_err = wrap_angle(raw_angle - self.env.target_angle)
        else:
            angle_err = wrap_angle(self.env.target_angle - raw_angle)

        velocity = float(self.env.data.qvel[0])
        control_torque = float(self.env.data.ctrl[0])
        reward = float(self.locals["rewards"][0]) if "rewards" in self.locals else 0.0
        t = time.time() - self.start_time

        print(
            f"时间: {t:.2f}s, raw_angle: {raw_angle:.3f} rad, 误差: {angle_err:.3f} rad ({np.degrees(angle_err):.2f}°), "
            f"速度: {velocity:.3f} rad/s, 力矩: {control_torque:.3f} Nm, 奖励: {reward:.3f}"
        )

        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps > 0:
            save_path = os.path.join(self.save_dir, f"ppo_model_{self.num_timesteps}_steps.zip")
            self.model.save(save_path)
            if self.verbose > 0:
                print(f"[Callback] 模型已保存到: {save_path}")

        self.viewer.sync()
        return True

    def _on_training_end(self) -> None:
        print("训练完成，viewer窗口保持打开（需手动关闭）")
        while self.viewer.is_running():
            time.sleep(0.01)


class BestModelSaverCallback(BaseCallback):
    def __init__(self, save_dir="./models", verbose=0):
        super().__init__(verbose)
        self.best_mean_reward = -np.inf
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if "ep_info_buffer" in self.locals and len(self.locals["ep_info_buffer"]) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.locals["ep_info_buffer"]])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_path = os.path.join(self.save_dir, "best_model.zip")
                self.model.save(best_path)
                if self.verbose > 0:
                    print(f"[BestModelSaver] 新最佳模型已保存: {best_path} 平均奖励: {mean_reward:.2f}")
        return True


class MetricsSaverCallback(BaseCallback):

    def __init__(self, save_dir="./metrics", verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.actor_losses = []
        self.critic_losses = []
        self.rewards = []

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0]) if "rewards" in self.locals else 0.0
        self.rewards.append(reward)
        return True

    def _on_rollout_end(self) -> None:
        clip_range = self.model.clip_range(self.model._current_progress_remaining)
        for rollout_data in self.model.rollout_buffer.get():
            obs = rollout_data.observations
            actions = rollout_data.actions
            returns = rollout_data.returns.flatten()
            old_values = rollout_data.old_values.flatten()
            advantages = rollout_data.advantages.flatten()
            old_log_prob = rollout_data.old_log_prob.flatten()

            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs)
                log_prob = dist.log_prob(actions).sum(axis=-1)
                values = self.model.policy.predict_values(obs).flatten()

            ratio = torch.exp(log_prob - old_log_prob)
            policy_loss_1 = ratio * advantages
            policy_loss_2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            value_loss = torch.nn.functional.mse_loss(values, returns)

            self.actor_losses.append(policy_loss.item())
            self.critic_losses.append(value_loss.item())

    def _on_training_end(self) -> None:
        np.savetxt(os.path.join(self.save_dir, "actor_loss.txt"), np.array(self.actor_losses))
        np.savetxt(os.path.join(self.save_dir, "critic_loss.txt"), np.array(self.critic_losses))
        np.savetxt(os.path.join(self.save_dir, "reward.txt"), np.array(self.rewards))
        if self.verbose > 0:
            print(f"[MetricsSaver] 已保存 Actor、Critic、Reward 曲线到 {self.save_dir}")


class InvertedPendulumEnv(gym.Env):
    def __init__(self, model_path, target_angle=0.0, clockwise_positive=True, angle_weight=1.0, vel_weight=0.1):
        super().__init__()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.target_angle = float(target_angle)
        self.clockwise_positive = bool(clockwise_positive)
        self.angle_weight = float(angle_weight)
        self.vel_weight = float(vel_weight)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)

        self.last_action = 0.0
        self.max_torque_rate = 0.1

    def _angle_error(self, raw_angle):
        if self.clockwise_positive:
            return wrap_angle(raw_angle - self.target_angle)
        else:
            return wrap_angle(self.target_angle - raw_angle)

    def _get_obs(self):
        raw_angle = float(self.data.qpos[0])
        angle_err = self._angle_error(raw_angle)
        vel = float(self.data.qvel[0])
        return np.array([np.cos(angle_err), np.sin(angle_err), vel], dtype=np.float32)

    def step(self, action):
        target_action = float(np.clip(action[0], -2.0, 2.0))
        delta = np.clip(target_action - self.last_action, -self.max_torque_rate, self.max_torque_rate)
        smoothed_action = self.last_action + delta
        self.last_action = smoothed_action

        self.data.ctrl[0] = float(smoothed_action)
        mujoco.mj_step(self.model, self.data)

        raw_angle = float(self.data.qpos[0])
        angle_err = self._angle_error(raw_angle)
        vel = float(self.data.qvel[0])
        reward = - (self.angle_weight * (angle_err ** 2) + self.vel_weight * (vel ** 2))

        obs = self._get_obs()
        return obs, float(reward), False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        init_perturb = self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[0] = float(self.target_angle + init_perturb)
        self.data.qvel[0] = 0.0
        self.last_action = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def render(self):
        pass