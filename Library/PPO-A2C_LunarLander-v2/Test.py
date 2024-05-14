import gym
from stable_baselines3 import A2C, PPO

# MODEL_DIR = "Models/A2C"
MODEL_DIR = "Models/PPO"

env = gym.make('LunarLander-v2')
env.reset()

model_path = f"{MODEL_DIR}/best_model.zip"
model = PPO.load(model_path, env=env)

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

env.close()
