import gym
from stable_baselines3 import PPO
import os

models_dir = "models/PPO"
model_path = f"{models_dir}/2300000.zip"

env = gym.make("LunarLander-v2")
env.reset()

model = PPO.load(model_path, env=env)

episodes = 20
for episode in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

env.close()