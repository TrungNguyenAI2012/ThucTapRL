import os
import gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# MODEL_DIR = "Models/DQN/"
MODEL_DIR = "Models/PPO/"
# MODEL_DIR = "Models/A2C/"
LOG_DIR = "Logs"

ENVIRONMENT_NAME = 'CartPole-v0'

env = gym.make(ENVIRONMENT_NAME)
env = DummyVecEnv([lambda: env])

# Callback
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eva_callback = EvalCallback(env,
                            callback_on_new_best=stop_callback,
                            eval_freq=10000,
                            verbose=1,
                            best_model_save_path=MODEL_DIR,
                            )

# Model
# model = DQN('MlpPolicy',
model = PPO('MlpPolicy',
# model = A2C('MlpPolicy',
            env,
            verbose=1,
            tensorboard_log=LOG_DIR)

TIMESTEP = 100000
model.learn(total_timesteps=TIMESTEP, callback=eva_callback)

env.close()
