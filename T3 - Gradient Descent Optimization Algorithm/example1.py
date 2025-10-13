"""
Author: Yanzeng Li
Date: 2025-10-13
Description: 使用梯度下降拟合线性方程
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(42)
true_a, true_b = 2.5, 1.2
n_samples = 100
noise_level = 1

# 生成线性分布 y = ax + b 的数据
x = np.linspace(0, 10, n_samples)
y = true_a * x + true_b + np.random.normal(0, noise_level, n_samples)

# 梯度下降参数
learning_rate = 0.02
n_iterations = 500

a_history, b_history = [], []

# 初始化参数
a, b = 0.01, 0.01

# 运行梯度下降
for i in range(n_iterations):
    # 计算预测值和误差
    y_pred = a * x + b
    error = y_pred - y
    
    # 计算梯度
    grad_a = np.mean(error * x)
    grad_b = np.mean(error)
    
    # 更新参数
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    
    # 记录历史
    a_history.append(a)
    b_history.append(b)


print(f"优化结果：y = {a}x + {b}")


# 计算正规方程解（解析解）
X = np.vstack([x, np.ones(len(x))]).T
theta_normal = np.linalg.inv(X.T @ X) @ X.T @ y
a_normal, b_normal = theta_normal

print(f"解析解：y = {a_normal}x + {b_normal}")


"""
后面全部都是绘图和动画了，同学可以无视
"""
a_grid = np.linspace(0, 4, 50)
b_grid = np.linspace(-2, 4, 50)
A, B = np.meshgrid(a_grid, b_grid)
loss = np.zeros_like(A)

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        y_pred = A[i,j] * x + B[i,j]
        loss[i,j] = np.mean((y_pred - y)**2) / 2

fig = plt.figure(figsize=(15, 5))

# 图1: 数据点和拟合线
ax1 = fig.add_subplot(1, 3, 1)
ax1.scatter(x, y, c='blue', label='Data points')
line, = ax1.plot([], [], 'r-', linewidth=2, label='Current fit')
ax1.set_xlim(0, 10)
ax1.set_ylim(min(y)-1, max(y)+1)
ax1.set_title('Data and Linear Fit')
ax1.legend()
ax1.grid(True)

# 图2: 损失函数曲面
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot_surface(A, B, loss, cmap='viridis', alpha=0.7, rstride=2, cstride=2)
path = ax2.plot([], [], [], 'r-', linewidth=2)[0]
point = ax2.plot([], [], [], 'ro')[0]
ax2.set_xlabel('a')
ax2.set_ylabel('b')
ax2.set_zlabel('Loss')
ax2.set_title('Loss Function Surface')
ax2.view_init(elev=30, azim=45)

# 图3: 损失函数等高线图
ax3 = fig.add_subplot(1, 3, 3)
contour = ax3.contour(A, B, loss, levels=15, cmap='viridis')
ax3.clabel(contour, inline=True, fontsize=8)
path2 = ax3.plot([], [], 'r-', linewidth=2)[0]
point2 = ax3.plot([], [], 'ro')[0]
ax3.plot(a_normal, b_normal, 'g*', markersize=10, label='Normal Equation')
ax3.legend()
ax3.grid(True)

plt.tight_layout()

path_coords = []
for a_val, b_val in zip(a_history, b_history):
    a_idx = np.argmin(np.abs(a_grid - a_val))
    b_idx = np.argmin(np.abs(b_grid - b_val))
    path_coords.append((a_val, b_val, loss[b_idx, a_idx]))
path_coords = np.array(path_coords).T


def update(frame):
    current_a, current_b = a_history[frame], b_history[frame]    
    line.set_data(x, current_a * x + current_b)
    path.set_data(path_coords[0, :frame+1], path_coords[1, :frame+1])
    path.set_3d_properties(path_coords[2, :frame+1])
    point.set_data([current_a], [current_b])
    point.set_3d_properties([path_coords[2, frame]])
    path2.set_data(a_history[:frame+1], b_history[:frame+1])
    point2.set_data([current_a], [current_b])
    ax1.set_title(f'Iteration {frame}: y = {current_a:.2f}x + {current_b:.2f}')
    return line, path, point, path2, point2


ani = FuncAnimation(
    fig, 
    update, 
    frames=range(n_iterations),
    interval=20,
    blit=False,
    cache_frame_data=False,  # 禁用缓存
    repeat=False
)

plt.ion()
plt.show(block=True)

while True:
    plt.pause(0.1)
