from stable_baselines3 import PPO
from Environment import ShowerEnv
from stable_baselines3.common.evaluation import evaluate_policy

MODEL_DIR = "Models/PPO/"

env = ShowerEnv()
env.reset()

model = PPO.load(f'{MODEL_DIR}best_model.zip', env=env)

print(evaluate_policy(model, env, n_eval_episodes=10))

env.close()
