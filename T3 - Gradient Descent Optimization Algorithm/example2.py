"""
Author: Yanzeng Li
Date: 2025-10-13
Description: 使用梯度下降解决二分类问题（逻辑回归）
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_classification

np.random.seed(42)

# 生成二分类数据
n_samples = 100
X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                          n_clusters_per_class=1, random_state=42)
y = y.reshape(-1, 1)

# 添加偏置项
X = np.hstack([X, np.ones((n_samples, 1))])

# 梯度下降参数
learning_rate = 0.2
n_iterations = 500

# 初始化参数
theta = np.random.rand(3, 1)
theta_history = []
loss_history = []

# 定义sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 计算损失函数
def compute_loss(theta):
    z = X @ theta
    h = sigmoid(z)
    loss = -np.mean(y * np.log(h) + (1-y) * np.log(1-h))
    return loss

# 运行梯度下降
for i in range(n_iterations):
    # 计算预测值和误差
    z = X @ theta
    h = sigmoid(z)
    error = h - y
    
    # 计算梯度
    grad = X.T @ error / n_samples
    
    # 更新参数
    theta -= learning_rate * grad
    
    # 记录历史
    theta_history.append(theta.copy())
    loss_history.append(compute_loss(theta))

print(f"优化结果：theta = {theta.flatten()}")

"""
后面全部都是绘图和动画了
"""
# 创建网格用于绘制决策边界
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                     np.linspace(y_min, y_max, 50))

fig = plt.figure(figsize=(15, 5))

# 图1: 数据点和决策边界
ax1 = fig.add_subplot(1, 3, 1)
scatter = ax1.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='bwr', edgecolors='k', label='Data points')
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.set_title('Data and Decision Boundary')
ax1.grid(True)

# 图2: 损失函数曲面
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
theta0 = np.linspace(-5, 5, 50)
theta1 = np.linspace(-5, 5, 50)
T0, T1 = np.meshgrid(theta0, theta1)
loss = np.zeros_like(T0)
for i in range(T0.shape[0]):
    for j in range(T0.shape[1]):
        loss[i,j] = compute_loss(np.array([T0[i,j], T1[i,j], theta_history[0][2][0]]).reshape(3,1))
ax2.plot_surface(T0, T1, loss, cmap='viridis', alpha=0.7, rstride=2, cstride=2)
path = ax2.plot([], [], [], 'r-', linewidth=2)[0]
point = ax2.plot([], [], [], 'ro')[0]
ax2.set_xlabel('theta0')
ax2.set_ylabel('theta1')
ax2.set_zlabel('Loss')
ax2.set_title('Loss Function Surface')
ax2.view_init(elev=30, azim=45)

# 图3: 损失函数变化曲线
ax3 = fig.add_subplot(1, 3, 3)
loss_line, = ax3.plot([], [], 'b-', linewidth=2, label='Loss')
current_point = ax3.plot([], [], 'ro')[0]
ax3.set_xlim(0, n_iterations)
ax3.set_ylim(0, max(loss_history)*1.1)
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Loss')
ax3.set_title('Loss Curve')
ax3.grid(True)
ax3.legend()

plt.tight_layout()

# 准备路径数据
theta0_history = [t[0][0] for t in theta_history]
theta1_history = [t[1][0] for t in theta_history]
path_loss_history = [compute_loss(np.array([t0, t1, theta_history[0][2][0]]).reshape(3,1)) 
                    for t0, t1 in zip(theta0_history, theta1_history)]

def update(frame):
    current_theta = theta_history[frame]
    
    # 计算当前决策边界
    Z = sigmoid(current_theta[0]*xx + current_theta[1]*yy + current_theta[2])
    
    # 更新第一个子图
    ax1.clear()
    ax1.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='bwr', edgecolors='k', label='Data points')
    ax1.contour(xx, yy, Z, levels=[0.5], colors='red', linewidths=2)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_title(f'Iteration {frame}: θ0={current_theta[0][0]:.2f}, θ1={current_theta[1][0]:.2f}')
    ax1.grid(True)
    
    # 更新3D路径
    path.set_data(theta0_history[:frame+1], theta1_history[:frame+1])
    path.set_3d_properties(path_loss_history[:frame+1])
    point.set_data([theta0_history[frame]], [theta1_history[frame]])
    point.set_3d_properties([path_loss_history[frame]])
    
    # 更新损失曲线
    loss_line.set_data(range(frame+1), loss_history[:frame+1])
    current_point.set_data([frame], [loss_history[frame]])
    
    return scatter, path, point, loss_line, current_point

ani = FuncAnimation(
    fig, 
    update, 
    frames=range(n_iterations),
    interval=20,
    blit=False,
    cache_frame_data=False,
    repeat=False
)

plt.ion()
plt.show(block=True)

while True:
    plt.pause(0.1)