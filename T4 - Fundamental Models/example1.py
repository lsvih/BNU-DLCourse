import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from tqdm import trange

def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, data_home='./mnist_data')
    X = mnist.data.astype('float32') / 255.0  # 归一化到[0,1]
    y = mnist.target.astype('int32')
    return X, y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 定义MLP类
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)  # Kaiming初始化
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros(output_size)
        
    def forward(self, X):
        # 前向传播
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]  # 样本数量
        
        # 计算输出层误差
        delta2 = self.a2 - y
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0) / m
        
        # 计算隐藏层误差
        delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0) / m
        
        # 更新参数
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate):
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []
        
        for epoch in trange(epochs):
            # 小批量训练
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                # 前向传播
                output = self.forward(X_batch)
                
                # 反向传播和参数更新
                self.backward(X_batch, y_batch, learning_rate)
            
            # 计算训练集和验证集的损失和准确率
            train_output = self.forward(X_train)
            train_loss = self.cross_entropy_loss(train_output, y_train)
            train_acc = self.accuracy(train_output, y_train)
            
            val_output = self.forward(X_val)
            val_loss = self.cross_entropy_loss(val_output, y_val)
            val_acc = self.accuracy(val_output, y_val)
            
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        return train_loss_history, val_loss_history, train_acc_history, val_acc_history
    
    def cross_entropy_loss(self, output, y):
        m = y.shape[0]
        log_probs = -np.log(output[range(m), y.argmax(axis=1)])
        loss = np.sum(log_probs) / m
        return loss
    
    def accuracy(self, output, y):
        predictions = np.argmax(output, axis=1)
        labels = np.argmax(y, axis=1)
        return np.mean(predictions == labels)

    def visualize_predictions(self, X, y, num_samples=10):
        # 随机选择一些样本进行可视化
        indices = np.random.choice(range(len(X)), size=num_samples, replace=False)
        samples = X[indices]
        labels = np.argmax(y[indices], axis=1)
        
        # 获取预测结果
        predictions = self.forward(samples)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # 可视化
        plt.figure(figsize=(15, 6))
        for i in range(num_samples):
            plt.subplot(2, num_samples//2, i+1)
            plt.imshow(samples[i].reshape(28, 28), cmap='gray')
            plt.title(f"True: {labels[i]}\nPred: {predicted_labels[i]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

# 主程序
def main():
    # 加载和预处理数据
    X, y = load_mnist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    # 将标签转换为one-hot编码
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_val = lb.transform(y_val)
    y_test = lb.transform(y_test)
    
    # 设置超参数
    input_size = 784
    hidden_size = 128
    output_size = 10
    epochs = 50
    batch_size = 64
    learning_rate = 0.1
    
    # 创建和训练MLP
    mlp = MLP(input_size, hidden_size, output_size)
    train_loss, val_loss, train_acc, val_acc = mlp.train(
        X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate
    )
    
    # 测试模型
    test_output = mlp.forward(X_test)
    test_acc = mlp.accuracy(test_output, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.show()

    print("\nVisualizing predictions on test set:")
    mlp.visualize_predictions(X_test, y_test, num_samples=10)

if __name__ == "__main__":
    main()
