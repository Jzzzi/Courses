import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

import util

# MLP模型

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LSTM, self).__init__()

        hidden_dim_1 = 20
        hidden_dim_2 = 40
        hidden_dim_3 = 20

        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.outlayer = nn.Linear(hidden_dim_3, output_dim)
        
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.outlayer(x))
        return x
    
# 训练函数

def train(model, train_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        if epoch%100 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# 数据处理函数

def data_process(original_data):
    """数据处理函数

    参数:
    original_data -- DataFrame原始数据
    """

    days_before = 15
    days_after = 3
    feature_num = 8
    k = 0.8
    batch_size = len(original_data) - days_before - days_after - 2

    num_samples = len(original_data) - days_before - days_after - 2
    # 特征为前days_before天的数据
    x = np.zeros((num_samples, days_before, feature_num))
    for i in range(num_samples):
        x[i] = original_data.loc[i:i + days_before - 1, 'open':].values
    # 标签为days_after后的涨跌
    # y = np.zeros((num_samples, 1))
    # for i in range(num_samples):
    #     if (original_data.loc[i+days_before+days_after-1, 'close'] - original_data.loc[i+days_before-1, 'close']) > 0:
    #         y[i] = 1.0
    #     else:
    #         y[i] = 0.0
    # 标签为days_after后的收盘价
    y = np.zeros((num_samples, 1))
    for i in range(num_samples):
        y[i] = original_data.loc[i+days_before+days_after-1, 'close']

    # 归一化
    scaler_x = MinMaxScaler()
    x = scaler_x.fit_transform(x.reshape(-1, feature_num)).reshape(num_samples, days_before, feature_num)
    x = x.reshape(num_samples, days_before, feature_num)
    scaler_y = MinMaxScaler()
    y = scaler_y.fit_transform(y.reshape(-1, 1)).reshape(num_samples, 1)
    y = y.reshape(num_samples, 1)

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

    # 转换为数据集
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, x_train, y_train, x_test, y_test, scaler_x, scaler_y

# 原始数据获取函数

def get_original_data(name, start_date, end_date):
    """获取股票的原始数据

    参数:
    name -- 字符串，股票名称
    start_date -- 字符串，开始日期
    end_date -- 字符串，结束日期
    """

    # 获取标的代码
    code = util.get_security(name)
    # 保存价格
    util.get_price_to_csv(code, start_date, end_date)
    # 读取价格
    original_data = pd.read_csv(f'{code}.csv')
    # 标定第一列为日期
    original_data.columns.values[0] = 'date'
    return original_data

# ==================== 主函数 ======================

original_data = get_original_data('华能水电', '2018-01-01', '2024-05-08')    

(train_loader, x_train, y_train,
 x_test, y_test,
 scaler_x, scaler_y) = data_process(original_data)

print(f'训练集大小: {len(train_loader.dataset)}')

input_dim = 15*8
num_epochs = 10000

model = MLP(input_dim, 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model, train_loader, optimizer, criterion, num_epochs)

model.eval()
y_pred_test = model(x_test)
print(f'测试集loss: {criterion(y_pred_test, y_test).item()}')
y_pred_test = y_pred_test.detach().numpy()
y_pred_test = y_pred_test.reshape(-1, 1)
y_pred_test = scaler_y.inverse_transform(y_pred_test)
y_test = y_test.detach().numpy()
y_test = y_test.reshape(-1, 1)
y_test = scaler_y.inverse_transform(y_test)


y_pred_train = model(train_loader.dataset.tensors[0])
print(f'测试集loss: {criterion(y_pred_train, y_train).item()}')
y_pred_train = y_pred_train.detach().numpy()
y_pred_train = y_pred_train.reshape(-1, 1)
y_pred_train = scaler_y.inverse_transform(y_pred_train)
y_train = train_loader.dataset.tensors[1].numpy()
y_train = y_train.reshape(-1, 1)
y_train = scaler_y.inverse_transform(y_train)


plt.subplot(1, 2, 1)
plt.title('Test')
plt.plot(y_pred_test, label='Prediction')
plt.plot(y_test, label='Real')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Train')
plt.plot(y_pred_train, label='Prediction')
plt.plot(y_train, label='Real')
plt.legend()
plt.show()
