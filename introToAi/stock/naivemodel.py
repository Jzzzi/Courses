from jqdatasdk import *
from util import *
from datetime import datetime, timedelta
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 登陆授权
auth('18973738468','040725Liu')

# 获取标的
start_date = '2022-01-01'
current_date = datetime.now()
yesterday_date = current_date - timedelta(days=1)
end_date = yesterday_date.strftime('%Y-%m-%d')
name = '华能水电'
code = get_security(name)

# 获取价格保存到csv文件
# get_price_to_csv(code, start_date, end_date)

# 从csv文件中读取数据
price = pd.read_csv(f'{code}.csv')
# 处理数据用于训练，用前days_before天的所有数据（包括今天）预测days_after天后相对今天的涨跌
# print(price.info())
days_before = 15
days_after = 1

open = price['open'].values
close = price['close'].values
high = price['high'].values
low = price['low'].values
volume = price['volume'].values
money = price['money'].values
total_len = len(open)

# 训练集和测试集
train_len = int(total_len * 0.7)
test_len = total_len - train_len

# 训练集
train_open = open[:train_len]
train_close = close[:train_len]
train_high = high[:train_len]
train_low = low[:train_len]
train_volume = volume[:train_len]
train_money = money[:train_len]

# 将其变为可供LSTM使用的输入train_input = (sequence_length, batch_size, feature_size)
# train_labels = (batch_size)
# 预测目标是days_after天后的涨跌率
sequence_length = days_before
batch_size = train_len - days_after - days_before
feature_size = 6
train_inputs = np.zeros((sequence_length, batch_size, feature_size))
train_labels = np.zeros(batch_size)
for i in range(batch_size):
    train_inputs[:, i, 0] = train_open[i:i+sequence_length]
    train_inputs[:, i, 1] = train_close[i:i+sequence_length]
    train_inputs[:, i, 2] = train_high[i:i+sequence_length]
    train_inputs[:, i, 3] = train_low[i:i+sequence_length]
    train_inputs[:, i, 4] = train_volume[i:i+sequence_length]
    train_inputs[:, i, 5] = train_money[i:i+sequence_length]
    new_price = train_close[i+sequence_length+days_after]
    old_price = train_close[i+sequence_length]
    if new_price > old_price:
        train_labels[i] = 1
    else:
        train_labels[i] = 0
# 归一化各个特征，并为之后反归一化做准备
feature_max = np.zeros(feature_size)
for i in range(feature_size):
    feature_max[i] = np.max(train_inputs[:, :, i])
    train_inputs[:, :, i] = train_inputs[:, :, i] / feature_max[i]

# LSTM模型

# 检查PyTorch是否支持GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class LSTMModel(nn.Module):
    def __init__(self, feature_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_size = output_size

        self.fc_1 = nn.Linear(self.feature_size, self.hidden_size_1)
        self.fc_2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.lstm = nn.LSTM(self.hidden_size_2, self.hidden_size_3)
        self.fc_3 = nn.Linear(self.hidden_size_3, self.output_size)


    def forward(self, x):
        batch_size = x.shape[1]
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(1, batch_size, self.hidden_size_3).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_size_3).to(device)

        # 前向传播
        # 假设x.shape = (sequence_length, batch_size, feature_size)
        sequence_length, batch_size, feature_size = x.shape
        # 展平x的前两个维度
        x_flattened = x.view(batch_size * sequence_length, feature_size)
        # 通过前两个全连接层映射到新的特征空间
        layer_1 = torch.relu(self.fc_1(x_flattened))
        layer_2 = torch.sigmoid(self.fc_2(layer_1))
        # 展平layer_1回到三维张量
        layer_2 = layer_2.view(sequence_length, batch_size, self.hidden_size_2)
        # LSTM层
        lstm_out, _ = self.lstm(layer_2, (h0, c0))
        # 只取最后一个时间步的输出
        lstm_out = lstm_out[-1, :, :]
        # lstm_ot.shape = (batch_size, hidden_size_3)
        # 经过最后一个全连接层并使用sigmoid激活函数
        output = torch.sigmoid(self.fc_3(lstm_out))
        output = output.view(batch_size)
        return output
    
# 模型参数
hidden_size_1 = 8
hidden_size_2 = 6
hidden_size_3 = 6
output_size = 1
# 实例化模型
model = LSTMModel(feature_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size).to(device)
# 损失函数和优化器
# 二分类问题使用交叉熵损失函数
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10000
train_inputs = torch.tensor(train_inputs, dtype=torch.float32).to(device)
train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)
# 训练模型
for epoch in range(num_epochs):  # num_epochs是训练轮数
    # 前向传播
    outputs = model(train_inputs)
    loss = criterion(outputs, train_labels)
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 在测试集上测试模型
test_open = open[train_len:]
test_close = close[train_len:]
test_high = high[train_len:]
test_low = low[train_len:]
test_volume = volume[train_len:]
test_money = money[train_len:]

# 将其变为可供LSTM使用的输入test_input = (sequence_length, batch_size, feature_size)
# test_labels = (batch_size)
# 预测目标是days_after天后的涨跌率
batch_size = test_len - days_after - days_before
test_inputs = np.zeros((sequence_length, batch_size, feature_size))
test_labels = np.zeros(batch_size)
for i in range(batch_size):
    test_inputs[:, i, 0] = test_open[i:i+sequence_length]
    test_inputs[:, i, 1] = test_close[i:i+sequence_length]
    test_inputs[:, i, 2] = test_high[i:i+sequence_length]
    test_inputs[:, i, 3] = test_low[i:i+sequence_length]
    test_inputs[:, i, 4] = test_volume[i:i+sequence_length]
    test_inputs[:, i, 5] = test_money[i:i+sequence_length]
    new_price = test_close[i+sequence_length+days_after]
    old_price = test_close[i+sequence_length]
    if new_price > old_price:
        test_labels[i] = 1
    else:
        test_labels[i] = 0
for i in range(feature_size):
    test_inputs[:, :, i] = test_inputs[:, :, i] / feature_max[i]
test_inputs = torch.tensor(test_inputs, dtype=torch.float32).to(device)
test_labels = torch.tensor(test_labels, dtype=torch.float32).to(device)
outputs = model(test_inputs)
loss = criterion(outputs, test_labels)
print(f'Test Loss: {loss.item()}')
# 输出结果
for i in range(batch_size):
    print(f'Predict: {outputs[i]}, Real: {test_labels[i]}')
# 保存模型
torch.save(model.state_dict(), 'model.pth')
print('Model has been saved')