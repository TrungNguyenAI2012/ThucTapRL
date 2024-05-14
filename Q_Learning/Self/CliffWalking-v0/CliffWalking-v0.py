import random
import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use("dark_background")

##################################################################################

# Hyperparameter
EPISODES = 10000
LEARNING_RATE = 0.2
DISCOUNT = 0.95
MAX_STEP = 500

# Khai phá môi trường
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = 0.01
epsilon_decay_value = 0.001

# Đếm số lần thành công
win_count = 0
# Lưu kết quả
reward_list = []
# Lưu kết quả tốt nhất
max_reward = -999
best_action_list = []

# Khởi tạo môi trường
env = gym.make('CliffWalking-v0')

# Bảng q_table (Nếu có lưu thì nhập đường dẫn)
# start_q_table = None
start_q_table = 'CliffWalking.pickle'

# Nếu chưa có q_table, tạo mới
if start_q_table is None:
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
else:
     with open(start_q_table, "rb") as f:
          q_table = pickle.load(f)

##################################################################################

# Q_Learning
for episode in range(EPISODES):
    state = env.reset()
    done = False
    ep_reward = 0
    ep_action_list = []

    for step in range(MAX_STEP):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > episode:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        ep_action_list.append(action)
        q_table[state, action] = (1 - LEARNING_RATE) * q_table[state, action] + LEARNING_RATE * (reward + DISCOUNT * np.max(q_table[next_state, :]))
        state = next_state
        ep_reward += reward

        if done:
            print('Hoàn thành tại {} với {} điểm'.format(episode + 1, ep_reward))
            win_count += 1
            if max_reward < ep_reward:
                print('=> Cập nhập lại!')
                max_reward = ep_reward
                best_action_list = ep_action_list
            break

    reward_list.append(ep_reward)

    epsilon = END_EPSILON_DECAYING + (START_EPSILON_DECAYING - END_EPSILON_DECAYING) * np.exp(-epsilon_decay_value * episode)

# Thống kê và xuất lần đặt điểm cao nhất
print('Hoàn thành', win_count, 'lần trong', EPISODES, 'lần thực hiện chiếm', win_count / EPISODES * 100, '%')
print('Điểm cao nhất', max_reward)
state = env.reset()
for action in best_action_list:
    state, reward, done, info = env.step(action)
    env.render()

# Tính điểm trung bình mỗi 1000 lần học
moving_avg = np.convolve(reward_list, np.ones((1000,))/1000, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Phần thưởng trung bình mỗi 1000 lần học")
plt.xlabel("Số lần học")
plt.show()

# Lưu mô hình
with open(f"CliffWalking.pickle", "wb") as f:
    pickle.dump(q_table, f)

env.close()
