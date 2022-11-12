from stable_baselines3 import PPO
import gym
import os

models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make("LunarLander-v2")

env.reset()
model = PPO("MlpPolicy", verbose=1, env=env, tensorboard_log=logdir)

TIMESTEPS = 100_000
for i in range(1, 100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f'{models_dir}/{TIMESTEPS*i}')

env.close()