from jqdatasdk import *
from util import *
from datetime import datetime, timedelta
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def price2data(price, fields, days_before, days_after, k):
    '''
    处理数据用于训练
    原始数据fields = ['open', 'close', 'high', 'low', 'volume', 'money', 'pre_close', 'avg']
    用days_before天的数据预测days_after天后的涨跌
    特征提取：
    0. 相比前日涨跌 = (close - pre_close) / pre_close
    1. 量比 = volume / volume_yesterday
    2. 额比 = money / money_yesterday
    3. 日内涨跌 = (close - open) / open
    4. 振幅 = (high - low) / low
    5. 均价比 = avg / avg_yesterday
    6. 开盘/前日收盘 = open / pre_close
    7. 当日收盘价
    8. 当日开盘价
    9. 当日最高价
    10. 当日最低价
    11. 当日成交量
    12. 当日成交额
    13. 当日均价
    data的形状为(num_samples, sequence_length, num_features)
    '''
    sequence_length = days_before
    feature_size = 14
    # 创造一个字典将列名和列索引对应
    column_index = {}
    for i in range(len(fields)):
        column_index[fields[i]] = i
    num_samples = price.shape[0]
    #处理得到训练输入和训练标签
    data = np.zeros((num_samples, sequence_length, feature_size))
    labels = np.zeros(num_samples)
    for i in range(num_samples):
        for j in range(sequence_length):
            if i-j >= 0:
                data[i, j, 0] = (price[i-j, column_index['close']] - price[i-j, column_index['pre_close']])/price[i-j, column_index['pre_close']]
                data[i, j, 1] = price[i-j, column_index['volume']] / price[i-j-1, column_index['volume']]
                data[i, j, 2] = price[i-j, column_index['money']] / price[i-j-1, column_index['money']]
                data[i, j, 3] = (price[i-j, column_index['close']] - price[i-j, column_index['open']]) / price[i-j, column_index['open']]
                data[i, j, 4] = (price[i-j, column_index['high']] - price[i-j, column_index['low']]) / price[i-j, column_index['low']]
                data[i, j, 5] = price[i-j, column_index['avg']] / price[i-j-1, column_index['avg']]
                data[i, j, 6] = price[i-j, column_index['open']] / price[i-j, column_index['pre_close']]
                data[i, j, 7] = price[i-j, column_index['close']]
                data[i, j, 8] = price[i-j, column_index['open']]
                data[i, j, 9] = price[i-j, column_index['high']]
                data[i, j, 10] = price[i-j, column_index['low']]
                data[i, j, 11] = price[i-j, column_index['volume']]
                data[i, j, 12] = price[i-j, column_index['money']]
                data[i, j, 13] = price[i-j, column_index['avg']]
        if i+days_after < num_samples:
            if price[i+days_after, column_index['close']] - price[i, column_index['close']] > 0:
                labels[i] = 1
            else:
                labels[i] = 0
        # if i+days_after < num_samples:
        #     labels[i] = price[i+days_after, column_index['close']] 
    # 归一化
    feature_norm = np.zeros(feature_size)
    feature_base = np.zeros(feature_size)
    label_norm = np.max(labels) - np.min(labels)
    label_base = np.min(labels)
    for i in range(feature_size):
        feature_norm[i] = np.max(data[:, :, i]) - np.min(data[:, :, i])
        feature_base[i] = np.min(data[:, :, i])
        data[:, :, i] = (data[:, :, i] - feature_base[i])/feature_norm[i]
    labels = (labels - label_base) / label_norm
    train_data = data[:int(k*num_samples)]
    train_labels = labels[:int(k*num_samples)]
    test_data = data[int(k*num_samples):]
    test_labels = labels[int(k*num_samples):]
    return train_data, train_labels, test_data, test_labels, num_samples, feature_size, feature_norm, feature_base, label_norm, label_base

class LSTMNet(nn.Module):
    '''
    定义LSTM网络
    '''
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNet, self).__init__()
        
        # 定义LSTM层
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)        
        # 定义输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # 解码最后一个时间步的隐藏状态
        out = self.fc(out[:, -1, :])
        out = torch.sigmoid(out)
        # 展平输出
        out = out.reshape(out.shape[0])
        return out


####################################################
#                   主程序                          #
####################################################
# 获取标的代码
start_date = '2023-01-03'
current_date = datetime.now()
yesterday_date = current_date - timedelta(days=1)
end_date = yesterday_date.strftime('%Y-%m-%d')
name = '创业50ETF'
code = get_security(name)
# code = '600025.XSHG'
fields = ['open', 'close', 'high', 'low', 'volume', 'money', 'pre_close', 'avg']

# 存储原始数据到csv文件
get_price_to_csv(code, start_date, end_date, fields = fields)
# raise DebugStop

# 读取原始数据
price = pd.read_csv(f'{code}.csv')
price = price.values
price = price[:, 1:] # 去掉第一列
days_before = 5 #序列长度
days_after = 1 # 预测天数
k = 0.7 # 训练集占比
train_data, train_labels, test_data, test_labels, num_samples, feature_size, feature_norm, feature_base, label_norm, label_base = price2data(price, fields, days_before, days_after, k)

# 超参数设置
input_size = feature_size  # 输入特征的维度
hidden_size = 16  # 隐藏层的维度
num_layers = 1  # LSTM层的数量
output_size = 1  # 输出的维度
batch_size = 16  # 批次大小


# 实例化LSTM网络
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
lstm = LSTMNet(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)
# lstm.load_state_dict(torch.load('lstm.pth'))

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(lstm.parameters(), lr=0.001)

############################################################################################################
# 训练模型
num_epochs = 200
for epoch in range(num_epochs):
    batches = batchify_sequences(train_data, train_labels, batch_size)
    for inputs, labels in batches:
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        
        # 前向传播
        outputs = lstm(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 保存模型
torch.save(lstm.state_dict(), 'lstm.pth')

###############################################################################################


# 训练集正确率
outputs = lstm(torch.tensor(train_data, dtype=torch.float32).to(device))
correct = 0
for i in range(len(train_labels)):
    if train_labels[i] > 0.5 and outputs[i] > 0.5:
        correct += 1
    elif train_labels[i] <= 0.5 and outputs[i] <= 0.5:
        correct += 1
accuracy = correct / len(train_labels)
print(f'Train Accuracy: {accuracy}')
# 测试集正确率
outputs = lstm(torch.tensor(test_data, dtype=torch.float32).to(device))
correct = 0
for i in range(len(outputs)):   
    if outputs[i] > 0.5 and test_labels[i] > 0.5:
        correct += 1
    elif outputs[i] <= 0.5 and test_labels[i] <= 0.5:
        correct += 1
accuracy = correct / len(outputs)
print(f'Test Accuracy: {accuracy}')

# 绘制预测结果
plt.plot(outputs.cpu().detach().numpy(), label='Predicted')
plt.plot(test_labels, label='True')
plt.legend()
plt.show()