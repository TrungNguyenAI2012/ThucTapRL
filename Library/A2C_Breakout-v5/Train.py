from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os

MODEL_DIR = os.path.join('Models', 'A2C')
LOG_DIR = os.path.join('Logs')

ENVIRONMENT_NAME = "ALE/Breakout-v5"
env = make_atari_env(ENVIRONMENT_NAME, n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)
env.reset()

# Callback
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=30, verbose=1)
eva_callback = EvalCallback(env,
                            callback_on_new_best=stop_callback,
                            eval_freq=10000,
                            verbose=1,
                            best_model_save_path=MODEL_DIR,
                            )

model = A2C("CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            )

model.learn(total_timesteps=100000, callback=eva_callback)

env.close()
