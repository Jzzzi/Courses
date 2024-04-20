from jqdatasdk import *
from util import *
from datetime import datetime, timedelta
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 获取标的代码
start_date = '2022-01-01'
current_date = datetime.now()
yesterday_date = current_date - timedelta(days=1)
end_date = yesterday_date.strftime('%Y-%m-%d')
name = '城市传媒'
code = get_security(name)

fields = ['open', 'close', 'high', 'low', 'volume', 'money', 'pre_close', 'avg']
# 存储原始数据到csv文件
get_price_to_csv(code, start_date, end_date, fields = fields)

# 读取原始数据
price = pd.read_csv(f'{code}.csv')
price = price.values
'''
列标索引:
0. 日期
1. 开盘价
2. 收盘价
3. 最高价
4. 最低价
5. 成交量
6. 成交额
7. 前收盘价
8. 均价
'''

# 创造一个字典将列名和列索引对应
column_index = {}
for i in range(len(fields)):
    column_index[fields[i]] = i + 1

# 处理数据用于训练
'''
基本想法:利用days_before天前(包括今天)的数据预测days_after天后的涨跌
特征提取:
0. 日内涨跌幅
1. 日内振幅
2. 与前日成交量比
3. 与前日成交额比
4. 与前日涨跌幅
5. 与前日振幅比
6. 与前日均价比
'''
days_before = 15
days_after = 5
total_len = len(price)
train_len = int(total_len * 0.5)
test_len = total_len - train_len - days_after - days_before
sequence_length = days_before
feature_size = 7

# 生成训练集
batch_size = train_len - days_before - days_after
train_inputs = np.zeros((sequence_length, batch_size, feature_size))
train_labels = np.zeros(batch_size)
for i in range(batch_size):
    for j in range(sequence_length):
        train_inputs[j, i, 0] = (price[i+j, column_index['close']] - price[i+j, column_index['open']])/price[i+j, column_index['open']]
        train_inputs[j, i, 1] = (price[i+j, column_index['high']] - price[i+j, column_index['low']])/price[i+j, column_index['low']]
        if i+j > 0:
            train_inputs[j, i, 2] = price[i+j, column_index['volume']]/price[i+j-1, column_index['volume']]
            train_inputs[j, i, 3] = price[i+j, column_index['money']]/price[i+j-1, column_index['money']]
            train_inputs[j, i, 4] = (price[i+j, column_index['close']] - price[i+j-1, column_index['close']])/price[i+j-1, column_index['close']]
            train_inputs[j, i, 5] = (price[i+j, column_index['high']] - price[i+j, column_index['low']])/price[i+j, column_index['low']]
            train_inputs[j, i, 6] = price[i+j, column_index['avg']]/price[i+j-1, column_index['avg']]
# 归一化
feature_max = np.zeros(feature_size)
for i in range(feature_size):
    feature_max[i] = np.max(train_inputs[:, :, i])
    train_inputs[:, :, i] = train_inputs[:, :, i] / feature_max[i]
# 变成张量
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)

# 定义LSTM模型
# 检查PyTorch是否支持GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class LSTMModel(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(self.feature_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, x):
        batch_size = x.shape[1]
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(device)

        # 前向传播
        # 假设x.shape = (sequence_length, batch_size, feature_size)
        sequence_length, batch_size, feature_size = x.shape
        # LSTM层
        lstm_out, _ = self.lstm(x, (h0, c0))
        # 只取最后一个时间步的输出
        lstm_out = lstm_out[-1, :, :]
        # lstm_ot.shape = (batch_size, hidden_size_3)
        # 经过最后一个全连接层并使用sigmoid激活函数
        output = torch.sigmoid(self.fc(lstm_out))
        output = output.view(batch_size)
        return output
    
# 模型参数
hidden_size = 8
output_size = 1
# 实例化模型
model = LSTMModel(feature_size, hidden_size, output_size).to(device)
# 损失函数和优化器
# 二分类问题使用交叉熵损失函数
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 2000
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
# 生成测试集
batch_size = test_len - days_before - days_after
test_inputs = np.zeros((sequence_length, batch_size, feature_size))
test_labels = np.zeros(batch_size)
for i in range(batch_size):
    for j in range(sequence_length):
        test_inputs[j, i, 0] = (price[train_len+i+j, column_index['close']] - price[train_len+i+j, column_index['open']])/price[train_len+i+j, column_index['open']]
        test_inputs[j, i, 1] = (price[train_len+i+j, column_index['high']] - price[train_len+i+j, column_index['low']])/price[train_len+i+j, column_index['low']]
        if i+j > 0:
            test_inputs[j, i, 2] = price[train_len+i+j, column_index['volume']]/price[train_len+i+j-1, column_index['volume']]
            test_inputs[j, i, 3] = price[train_len+i+j, column_index['money']]/price[train_len+i+j-1, column_index['money']]
            test_inputs[j, i, 4] = (price[train_len+i+j, column_index['close']] - price[train_len+i+j-1, column_index['close']])/price[train_len+i+j-1, column_index['close']]
            test_inputs[j, i, 5] = (price[train_len+i+j, column_index['high']] - price[train_len+i+j, column_index['low']])/price[train_len+i+j, column_index['low']]
            test_inputs[j, i, 6] = price[train_len+i+j, column_index['avg']]/price[train_len+i+j-1, column_index['avg']]
    if price[train_len+i+sequence_length+days_after, column_index['close']] > price[train_len+i+sequence_length, column_index['close']]:
        test_labels[i] = 1
# 归一化
for i in range(feature_size):
    test_inputs[:, :, i] = test_inputs[:, :, i] / feature_max[i]
# 变成张量
test_inputs = torch.tensor(test_inputs, dtype=torch.float32).to(device)
# 预测
outputs = model(test_inputs)
# 计算准确率
correct = 0
for i in range(batch_size):
    if outputs[i] > 0.5:
        if test_labels[i] == 1:
            correct += 1
    else:
        if test_labels[i] == 0:
            correct += 1
accuracy = correct / batch_size
print(f'Accuracy: {accuracy}')

# 保存模型
torch.save(model.state_dict(), 'simplemodel.pth')

#计算收益率
bought = False
profit = 1
for i in range(batch_size):
    if outputs[i] > 0.5:
        if not bought:
            bought = True
            print('买了!')
            bought_price = price[train_len+i+sequence_length, column_index['close']]
    else:
        if bought:
            bought = False
            print('卖了!')
            sold_price = price[train_len+i+sequence_length, column_index['close']]
            profit *= sold_price / bought_price
if bought:
    sold_price = price[train_len+batch_size+sequence_length, column_index['close']]
    profit *= sold_price / bought_price
print(f'Profit: {profit}')
