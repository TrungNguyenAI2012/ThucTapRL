import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
style.use("dark_background")

##################################################################################

# Hyperparameter
# EPISODES = 25000
EPISODES = 10
LEARNING_RATE = 0.1
DISCOUNT = 0.95

# Cài đặt khai phá môi trường
# epsilon = 1
epsilon = 0
EPS_DECAY = 0.999

# Cài đặt môi trường
SIZE = 10  # Kích thước môi trường
MOVE_PENALTY = 1  # Điểm phạt di chuyển
ENEMY_PENALTY = 400  # Điểm phạt đụng kẻ thù
FOOD_REWARD = 30  # Điểm thưởng ăn được đồ ăn
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3
d = {1: (255, 0, 0),  # blue
     2: (0, 255, 0),  # green
     3: (0, 0, 255)}  # red

# Đếm số lần thành công
win_count = 0
# Lưu kết quả
reward_list = []
# Lưu kết quả tốt nhất
max_reward = -999
best_action_list = []

# Bảng q_table (Nếu có lưu thì nhập đường dẫn)
# start_q_table = None
start_q_table = 'Blob.pickle'

# Nếu chưa có q_table, tạo mới
if start_q_table is None:
     q_table = {}
     for x1 in range(-SIZE + 1, SIZE):
          for x2 in range(-SIZE + 1, SIZE):
               for y1 in range(-SIZE + 1, SIZE):
                    for y2 in range(-SIZE + 1, SIZE):
                         q_table[((x1, x2), (y1, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
     with open(start_q_table, "rb") as f:
          q_table = pickle.load(f)

##################################################################################

class Blob:
     # Tạo vị trí ngẫu nhiên
     def __init__(self):
          self.x = np.random.randint(0, SIZE)
          self.y = np.random.randint(0, SIZE)

     # Tránh bị trùng vị trí nhau
     def __str__(self):
          return f"{self.x}, {self.y}"

     # Phương thức ghi đè nhau (Nếu cùng vị trí)
     def __sub__(self, other):
          return (self.x - other.x, self.y - other.y)

     # Phương thức di chuyển của Agent
     def move(self, x=False, y=False):
          # Nếu không có x, di chuyển x ngẫu nhiên
          if not x:
               self.x += np.random.randint(-1, 2)
          else:
               self.x += x
          # Nếu không có y, di chuyển y ngẫu nhiên
          if not y:
               self.y += np.random.randint(-1, 2)
          else:
               self.y += y
          # Nếu di chuyển qua khu vực thì giữ lại
          if self.x < 0:
               self.x = 0
          elif self.x > SIZE - 1:
               self.x = SIZE - 1
          if self.y < 0:
               self.y = 0
          elif self.y > SIZE - 1:
               self.y = SIZE - 1

     # Thực hiện hành động
     def action(self, choice):
          if choice == 0:  # Up
               self.move(x=0, y=1)
          elif choice == 1:  # Down
               self.move(x=0, y=-1)
          elif choice == 2:  # Left
               self.move(x=-1, y=0)
          elif choice == 3:  # Right
               self.move(x=1, y=0)

##################################################################################

# Q_Learning
for episode in range(EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()
    ep_reward = 0
    ep_action_list = []

    for i in range(300):
         env = (player - food, player - enemy)
         exploration_rate_threshold = np.random.random()
         if exploration_rate_threshold > epsilon:
              action = np.argmax(q_table[env])
         else:
              action = np.random.randint(0, 4)
         player.action(action)

         # Cho thức ăn và kẻ thù di chuyển
         enemy.move()
         food.move()

         # Tính phần thưởng
         if player.x == enemy.x and player.y == enemy.y:
              reward = -ENEMY_PENALTY
         elif player.x == food.x and player.y == food.y:
              reward = FOOD_REWARD
         else:
              reward = -MOVE_PENALTY

         new_obs = (player - food, player - enemy)
         max_future_q = np.max(q_table[new_obs])
         current_q = q_table[env][action]
         if reward == FOOD_REWARD:
              new_q = FOOD_REWARD
         else:
              new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
         q_table[env][action] = new_q
         ep_reward += reward

         # Xuất ảnh
         # if episode % 1000 == 0:
         if episode % 1 == 0:
              env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
              env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
              env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue
              env[enemy.x][enemy.y] = d[ENEMY_N]  # sets the enemy location to red
              img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
              img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
              cv2.imshow("image", np.array(img))  # show it!
              if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                   if cv2.waitKey(1000) & 0xFF == ord('q'):
                        break
              else:
                   if cv2.waitKey(100) & 0xFF == ord('q'):
                        break

         if reward == FOOD_REWARD:
              win_count += 1
              print('Hoàn thành tại {} với {} điểm'.format(episode + 1, ep_reward))
              if max_reward < ep_reward:
                   print('=> Cập nhập lại!')
                   max_reward = ep_reward
                   best_action_list = ep_action_list
              break
         elif reward == -ENEMY_PENALTY:
              break

    reward_list.append(ep_reward)
    epsilon *= EPS_DECAY

# Thống kê và xuất lần đặt điểm cao nhất
print('Hoàn thành', win_count, 'lần trong', EPISODES, 'lần thực hiện chiếm', win_count / EPISODES * 100, '%')
print('Điểm cao nhất', max_reward)

# Tính điểm trung bình mỗi 1000 lần học
moving_avg = np.convolve(reward_list, np.ones((1000,))/1000, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Phần thưởng trung bình mỗi 1000 lần học")
plt.xlabel("Số lần học")
plt.show()

# Lưu mô hình
with open(f"Blob.pickle", "wb") as f:
    pickle.dump(q_table, f)
