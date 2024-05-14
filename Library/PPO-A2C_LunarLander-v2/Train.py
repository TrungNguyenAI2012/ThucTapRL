import gym
from stable_baselines3 import A2C, PPO
import os
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# MODEL_DIR = "Models/A2C/"
MODEL_DIR = "Models/PPO/"
LOG_DIR = "Logs"

env = gym.make('LunarLander-v2')
env.reset()

# Callback
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eva_callback = EvalCallback(env,
                            callback_on_new_best=stop_callback,
                            eval_freq=10000,
                            verbose=1,
                            best_model_save_path=MODEL_DIR,
                            )

# model = A2C('MlpPolicy'
model = PPO('MlpPolicy',
            env,
            verbose=1,
            tensorboard_log=LOG_DIR)

TIMESTEP = 1000000
model.learn(total_timesteps=TIMESTEP, callback=eva_callback)

env.close()
