from stable_baselines3 import PPO
from RL_Function_PPO import InvertedPendulumEnv, cosine_annealing_lr, ViewerCallback, BestModelSaverCallback, \
    MetricsSaverCallback

if __name__ == "__main__":
    model_path = r"YOUR_PATH\Model_Pendulum.xml"
    target_angle = 0.0
    clockwise_positive = True
    angle_weight = 0.7
    vel_weight = 0.3
    net_arch_actor = [1024, 512]
    net_arch_critic = [1024, 512]
    save_freq = 100000
    save_dir = "./models0816"
    initial_lr = 3e-4
    n_steps = 2048
    total_timesteps = 500000

    env = InvertedPendulumEnv(model_path, target_angle, clockwise_positive, angle_weight, vel_weight)
    policy_kwargs = dict(net_arch=[dict(pi=net_arch_actor, vf=net_arch_critic)])

    ppo_model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=cosine_annealing_lr(initial_lr, total_timesteps),
        n_steps=n_steps
    )

    print("开始在线训练PPO模型（viewer实时显示）...")
    callback = ViewerCallback(env, save_freq=save_freq, save_dir=save_dir, verbose=1)
    best_callback = BestModelSaverCallback(save_dir=save_dir, verbose=1)
    metrics_callback = MetricsSaverCallback(save_dir="./Loss_metrics", verbose=1)

    ppo_model.learn(total_timesteps=total_timesteps, callback=[callback, best_callback, metrics_callback])
