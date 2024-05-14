from stable_baselines3 import PPO
import os
from Environment import SnekEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

MODEL_DIR = "Models/PPO/"
LOG_DIR = "Logs/"

env = SnekEnv()
env.reset()

# Callback
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=100, verbose=1)
eva_callback = EvalCallback(env,
                            callback_on_new_best=stop_callback,
                            eval_freq=10000,
                            verbose=1,
                            best_model_save_path=MODEL_DIR,
                            )

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=LOG_DIR)

TIMESTEP = 1000000
model.learn(total_timesteps=TIMESTEP, callback=eva_callback)

env.close()
