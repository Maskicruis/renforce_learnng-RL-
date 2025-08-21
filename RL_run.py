import time
import mujoco
from stable_baselines3 import PPO
from RL_Function_PPO import InvertedPendulumEnv

if __name__ == "__main__":
    model_path = r"YOUR_PATH\Model_Pendulum.xml"
    saved_model_path = "your_path"  # 替换为训练好的模型路径

    env = InvertedPendulumEnv(model_path, target_angle=0.0, clockwise_positive=True, angle_weight=0.7, vel_weight=0.3)
    model = PPO.load(saved_model_path, env=env)

    viewer = mujoco.viewer.launch_passive(env.model, env.data)

    obs, _ = env.reset()  # 注意这里解包obs和info

    while viewer.is_running():
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

        raw_angle = float(env.data.qpos[0])
        angle_err = env._angle_error(raw_angle)
        velocity = float(env.data.qvel[0])
        control_torque = float(env.data.ctrl[0])

        print(f"角度: {raw_angle:.3f} rad, 误差: {angle_err:.3f} rad, 速度: {velocity:.3f} rad/s, 力矩: {control_torque:.3f} Nm, 奖励: {reward:.3f}")

        viewer.sync()
        time.sleep(0.01)

        if done:
            obs, _ = env.reset()
