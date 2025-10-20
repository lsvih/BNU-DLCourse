import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# 加载数据
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, data_home='./mnist_data')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int64')
    return torch.tensor(X), torch.tensor(y)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = correct_train / total_train
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct_val / total_val
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    return train_loss_history, val_loss_history, train_acc_history, val_acc_history

# 可视化预测结果
def visualize_predictions(model, test_loader, num_samples=10):
    model.eval()
    with torch.no_grad():
        # 获取一批测试数据
        inputs, labels = next(iter(test_loader))
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        # 可视化
        plt.figure(figsize=(15, 6))
        for i in range(num_samples):
            plt.subplot(2, num_samples//2, i+1)
            plt.imshow(inputs[i].reshape(28, 28), cmap='gray')
            plt.title(f"True: {labels[i].item()}\nPred: {predicted[i].item()}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

# 主程序
def main():
    # 加载数据
    X, y = load_mnist()
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    # 创建DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 模型参数
    input_size = 784
    hidden_size = 128
    output_size = 10
    epochs = 50
    learning_rate = 0.1
    
    # 初始化模型、损失函数和优化器
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # 训练模型
    train_loss, val_loss, train_acc, val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, epochs
    )
    
    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = correct / total
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
    
    # 可视化预测结果
    print("\nVisualizing predictions on test set:")
    visualize_predictions(model, test_loader, num_samples=10)

if __name__ == "__main__":
    main()
