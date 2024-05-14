import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace  # Import the Joypad wrapper
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv  # Import Vectorization Wrappers
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# 1. Create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# 2. Simplify the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')

MODEL_DIR = 'Models/PPO/'
LOG_DIR = 'Logs'

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=1)
eva_callback = EvalCallback(env,
                            callback_on_new_best=stop_callback,
                            eval_freq=10000,
                            verbose=1,
                            best_model_save_path=MODEL_DIR,
                            )
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)

TIMESTEP = 100000000
model.learn(total_timesteps=TIMESTEP, callback=eva_callback)
