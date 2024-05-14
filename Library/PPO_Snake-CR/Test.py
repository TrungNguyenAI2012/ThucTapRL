from Environment import SnekEnv
from stable_baselines3 import PPO
import gym

env = SnekEnv()
env.reset()

model_dir = "Models/PPO/"
model = PPO.load(f"{model_dir}best_model.zip", env=env)

# Test model
for episode in range(10):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        score += reward
    print('Episode:{} => Score {}'.format(episode, score))

env.close()