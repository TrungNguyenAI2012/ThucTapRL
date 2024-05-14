from stable_baselines3 import A2C
import gym

MODEL_DIR = 'Models/A2C/'
LOG_DIR = 'Logs'

ENVIRONMENT_NAME = "ALE/Adventure-v5"
env = gym.make(ENVIRONMENT_NAME, render_mode='human')
env.reset()

model = A2C.load(f'{MODEL_DIR}best_model.zip', env=env)

for episode in range(10):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} => Score {}'.format(episode, score))

env.close()
