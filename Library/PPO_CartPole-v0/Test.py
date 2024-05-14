import gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# MODEL_DIR = "Models/DQN/"
MODEL_DIR = "Models/PPO/"
# MODEL_DIR = "Models/A2C/"

ENVIRONMENT_NAME = 'CartPole-v0'
EPISODES = 25000
env = gym.make(ENVIRONMENT_NAME)
env = DummyVecEnv([lambda: env])

# model = DQN.load(f'{MODEL_DIR}best_model.zip', env=env)
model = PPO.load(f'{MODEL_DIR}best_model.zip', env=env)
# model = A2C.load(f'{MODEL_DIR}best_model.zip', env=env)

evaluate_policy(model, env, n_eval_episodes=1, render=True)

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
