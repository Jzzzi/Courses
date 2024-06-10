import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import util

def dataProcess(original_data, days_before=15, days_after=5, batch_size=100, k=0.8):
    """数据处理函数

    参数:
    original_data -- DataFrame原始数据
    """
    print('Start data processing...')
    # 去除无效数据
    original_data = original_data[original_data['open'] != 0]
    original_data = original_data[original_data['high'] != 0]
    original_data = original_data[original_data['low'] != 0]
    original_data = original_data[original_data['close'] != 0]
    original_data = original_data[original_data['volume'] != 0]
    original_data = original_data[original_data['money'] != 0]
    original_data = original_data[original_data['pre_close'] != 0]
    original_data = original_data[original_data['avg'] != 0]
    # 去除无效数据后的索引重置
    original_data = original_data.reset_index(drop=True)

    # 样本总数
    num_samples = len(original_data) - days_before - days_after
    print(f'样本总数: {num_samples}')
    # 特征为前days_before天的数据\
    # numsamples 序列个数
    # days_before 序列长度
    # 8 特征数
    x = np.zeros((num_samples, days_before, 8))
    for i in range(num_samples):
        x[i] = original_data.loc[i+1:i + days_before, 'open':].values/original_data.loc[i:i + days_before-1, 'open':].values
        # x[i] = original_data.loc[i:i + days_before - 1, 'open':].values
    # 标签为days_after后的涨跌
    y = np.zeros((num_samples, 2))
    for i in range(num_samples):
        if (original_data.loc[i+days_before+days_after, 'close'] - original_data.loc[i+days_before, 'close']) > 0:
            y[i,1] = 1# [0, 1]
        else:
            y[i,0] = 1# [1, 0]

    # 归一化
    scaler_x = MinMaxScaler()
    x = scaler_x.fit_transform(x.reshape(-1, 8)).reshape(num_samples, days_before, 8)
    # 划分训练集和测试集
    train_size = int(num_samples * k)
    x_train = x[:train_size]
    y_train = y[:train_size]

    x_test = x[train_size:]
    y_test = y[train_size:]

    # 转换为张量
    x_train = torch.tensor(x_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    x_test = torch.tensor(x_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)
    # 如果使用GPU
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_test = x_test.cuda()
        y_test = y_test.cuda()

    # 转换为数据集
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, x_train, y_train, x_test, y_test, scaler_x

def train(model, train_loader, num_epochs=500, lr=0.001):
    """训练函数

    参数:
    model -- 网络模型
    train_loader -- 训练集
    num_epochs -- 训练轮数
    lr -- 学习率
    """
    print('Start training...')
    model.train()
    criterion = nn.CrossEntropyLoss() # 交错熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr) # 优化器
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_loader):
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%100 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x
    

if __name__ == '__main__':
    original_data = pd.read_csv('002765.XSHE.csv')
    #=========================HyperParameter
    days_before = 15
    days_after = 5
    batch_size = 100
    k = 0.9
    #=========================
    # 训练集占比
    num_samples = len(original_data) - days_before - days_after 
    train_loader, x_train, y_train, x_test, y_test, scaler_x = dataProcess(original_data,batch_size=batch_size,k=k,days_before=days_before,days_after=days_after)
    
    #=======================HyperParameter
    hidden_size = 6
    num_layers = 2
    dropout = 0
    #=======================
    model = LSTM(8, hidden_size=hidden_size, num_layers=num_layers, num_classes=2, dropout=dropout)

    # model = LSTM(8)
    if torch.cuda.is_available():
        model = model.cuda()

    # 训练模型
    #======================HyperParameter
    num_epochs = 1000
    lr = 0.005
    #======================
    # train(model, train_loader, num_epochs=num_epochs, lr=lr)
    # 读取模型
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('model.pth'))
    else:
        model.load_state_dict(torch.load('model.pth', map_location='cpu')) 
    model.eval()
    # 训练集准确率
    y_pred = model(x_train)
    _, predicted = torch.max(y_pred, 1)
    total = y_train.size(0)
    correct = 0
    for i in range(len(predicted)):
        if predicted[i] == torch.argmax(y_train[i]):
            correct += 1
    print(f'训练集准确率: {correct/total}')
    # 测试集准确率
    y_pred = model(x_test)
    _, predicted = torch.max(y_pred, 1)
    total = y_test.size(0)
    correct = 0
    for i in range(len(predicted)):
        if predicted[i] == torch.argmax(y_test[i]):
            correct += 1
    print(f'测试集准确率: {correct/total}')
    # 测试集上涨的概率
    total = y_test.size(0)
    correct = 0
    for i in range(len(predicted)):
        if y_test[i][1] == 1:
            correct += 1
    print(f'测试集上涨概率: {correct/total}')

    # 保存模型
    key = input('是否保存模型？(y/n)')
    if key == 'y':
        torch.save(model.state_dict(), 'model.pth')
        print('模型已保存')
    
    # 画出股票走势图
    import matplotlib.pyplot as plt
    original_data = original_data[days_before:]
    original_data = original_data.reset_index(drop=True)
    original_data = original_data.dropna()
    original_data['close'].plot()
    plt.show()