import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use("dark_background")

##################################################################################

# Hyperparameter
EPISODES = 100
LEARNING_RATE = 0.1
DISCOUNT = 0.95

# Cài đặt khai phá môi trường
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Đếm số lần thành công
win_count = 0
# Lưu kết quả
reward_list = []
# Lưu kết quả tốt nhất
max_reward = -999
max_start_state = None
best_action_list = []

# Khởi tạo môi trường
env = gym.make("MountainCar-v0")

# Bảng q_table (Nếu có lưu thì nhập đường dẫn)
# start_q_table = None
start_q_table = 'MountainCar.pickle'

# Nếu chưa có q_table, tạo mới
DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
if start_q_table is None:
    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
else:
    with open(start_q_table, "rb") as f:
      q_table = pickle.load(f)

##################################################################################

# Hàm chuyển state về discrete_state
# Tra cứu 3 giá trị Q cho các hành động có sẵn trong bảng q_table
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))

##################################################################################

# Q Learning
for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False
    ep_reward = 0
    ep_start_state = discrete_state
    ep_action_list = []

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = env.action_space.sample()
        ep_action_list.append(action)
        new_state, reward, done, _ = env.step(action)
        ep_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        if not done:
            # Giá trị Q tối đa có thể có trong bước tiếp theo (đối với trạng thái mới)
            max_future_q = np.max(q_table[new_discrete_state])
            # giá trị Q hiện tại (cho trạng thái hiện tại và hành động đã thực hiện)
            current_q = q_table[discrete_state + (action,)]
            # Q mới cho trạng thái và hành động hiện tại
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # Cập nhập q_table với giá trị Q mới
            q_table[discrete_state + (action,)] = new_q
        # Nếu mục tiêu tới đích - cập nhật trực tiếp giá trị Q với phần thưởng tức thời
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
            win_count += 1
            print('Hoàn thành tại {} với {} điểm'.format(episode + 1, ep_reward))
            if max_reward < ep_reward:
                print('=> Cập nhập lại!')
                max_reward = ep_reward
                best_action_list = ep_action_list
                max_start_state = ep_start_state
        discrete_state = new_discrete_state

    reward_list.append(ep_reward)

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

##################################################################################

# Thống kê và xuất lần đặt điểm cao nhất
print('Hoàn thành', win_count, 'lần trong', EPISODES, 'lần thực hiện chiếm', win_count / EPISODES * 100, '%')
print('Điểm cao nhất', max_reward)
state = max_start_state
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
with open(f"MountainCar.pickle", "wb") as f:
    pickle.dump(q_table, f)

env.close()
