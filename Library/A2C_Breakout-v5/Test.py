from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env

MODEL_DIR = 'Models/A2C/'
LOG_DIR = 'Logs'

ENVIRONMENT_NAME = "ALE/Breakout-v5"
env = make_atari_env(ENVIRONMENT_NAME, n_envs=4, seed=1)
env = VecFrameStack(env, n_stack=4)
env.reset()

model = A2C.load(f'{MODEL_DIR}best_model.zip', env=env)

evaluate_policy(model, env, n_eval_episodes=10, render=True)

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

# for episode in range(10):
#     obs = env.reset()
#     done = False
#     score = 0
#
#     while True:
#         env.render()
#         action, _ = model.predict(obs)
#         obs, reward, done, info = env.step(action)
#         env.render(mode='human')
#         score += reward
#     print('Episode:{} => Score {}'.format(episode, score))

env.close()
