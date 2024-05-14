from stable_baselines3 import PPO
from Environment import ShowerEnv
import os
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

env = ShowerEnv()
env.reset()

MODEL_DIR = 'Models/PPO/'
LOG_DIR = 'Logs'

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eva_callback = EvalCallback(env,
                            callback_on_new_best=stop_callback,
                            eval_freq=10000,
                            verbose=1,
                            best_model_save_path=MODEL_DIR,
                            )

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='Logs')
model.learn(total_timesteps=100000, callback=eva_callback)

env.close()
