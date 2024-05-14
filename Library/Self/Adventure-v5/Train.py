from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
import gym

MODEL_DIR = os.path.join('Models', 'A2C')
LOG_DIR = os.path.join('Logs')

ENVIRONMENT_NAME = "ALE/Adventure-v5"
env = gym.make(ENVIRONMENT_NAME)
env.reset()

# Callback
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)
eva_callback = EvalCallback(env,
                            callback_on_new_best=stop_callback,
                            eval_freq=10000,
                            verbose=1,
                            best_model_save_path=MODEL_DIR,
                            )

model = PPO("CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            )

model.learn(total_timesteps=100000, callback=eva_callback)

env.close()
