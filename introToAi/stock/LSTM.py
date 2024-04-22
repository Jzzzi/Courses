from jqdatasdk import *
from util import *
from datetime import datetime, timedelta
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
'''
############################################################################################################
获取股票数据
'''
# 获取标的代码
start_date = '2018-01-01'
current_date = datetime.now()
yesterday_date = current_date - timedelta(days=1)
end_date = yesterday_date.strftime('%Y-%m-%d')
name = '华能水电'
code = get_security(name)
fields = ['open', 'close', 'high', 'low', 'volume', 'money', 'pre_close', 'avg']

# 存储原始数据到csv文件
# get_price_to_csv(code, start_date, end_date, fields = fields)
# raise DebugStop

# 读取原始数据
price = pd.read_csv(f'{code}.csv')
price = price.values
price = price[:, 1:] # 去掉第一列

# 创造一个字典将列名和列索引对应
column_index = {}
for i in range(len(fields)):
    column_index[fields[i]] = i

'''
############################################################################################################
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
data的形状为(num_samples, sequence_length, num_features)
'''
days_before = 15
days_after = 3
sequence_length = days_before
feature_size = 7
k = 0.7
num_samples = k * price.shape[0]
#处理得到训练输入和训练标签
data = np.zeros((num_samples, sequence_length, feature_size))
for i in range(num_samples):
     for j in range(sequence_length):
            if i-j >= 0:
                data[i, j, 0] = (price[i-j, column_index['close']] - price[i-j, column_index['pre_close']])/price[i-j, column_index['pre_close']]
                data[i, j, 1] = price[i-j, column_index['volume']] / price[i-j-1, column_index['volume']]
                data[i, j, 2] = price[i-j, column_index['money']] / price[i-j-1, column_index['money']
                data[i, j, 3] = (price[i-j, column_index['close']] - price[i-j, column_index['open']]) / price[i-j, column_index['open']]
                data[i, j, 4] = (price[i-j, column_index['high']] - price[i-j, column_index['low']]) / price[i-j, column_index['low']]
                data[i, j, 5] = price[i-j, column_index['avg']] / price[i-j-1, column_index['avg']]
                data[i, j, 6] = price[i-j, column_index['open']] / price[i-j, column_index['pre_close']]
    
# 归一化
feature_norm = np.zeros(feature_size)
feature_base = np.zeros(feature_size)
for i in range(feature_size):
    feature_base[i] = np.min(train_inputs[:, :, i])
    feature_norm[i] = np.max(train_inputs[:, :, i]) - np.min(train_inputs[:, :, i])
    train_inputs[:, :, i] -= feature_base[i]
    train_inputs[:, :, i] /= feature_norm[i]
label_base = np.min(train_labels)
label_norm = np.max(train_labels) - np.min(train_labels)
train_labels -= label_base
train_labels /= label_norm
np.save('train_inputs.npy', train_inputs)
np.save('train_labels.npy', train_labels)
# 绘制训练标签
# plt.plot(train_labels)
# plt.show()
# raise DebugStop
# 检查PyTorch是否支持GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
'''
############################################################################################################
LSTM模型
'''
# 定义LSTM网络结构
class LSTMNet(nn.Module):
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
        return out

# 超参数设置
input_size = 10  # 输入特征的维度
hidden_size = 50  # 隐藏层的维度
num_layers = 1  # LSTM层的数量
output_size = 1  # 输出的维度

# 绘出训练标签和模型输出
outputs = model(train_inputs)
outputs = outputs.cpu().detach().numpy()
outputs = outputs * label_norm + label_base
train_labels = train_labels.cpu().detach().numpy()
train_labels = train_labels * label_norm + label_base
plt.plot(train_labels, label = 'train_labels')
plt.plot(outputs, label = 'outputs')
# 画出y=0的虚线
plt.axhline(y=0, color='r', linestyle='--')
plt.legend()
plt.show()
raise DebugStop
'''
############################################################################################################
模型测试部分
'''

#处理得到测试输入和测试标签
test_inputs = np.zeros((sequence_length, test_len, feature_size))
test_labels = np.zeros(test_len)
for i in range(train_len, price.shape[0]-days_after):
    for j in range(sequence_length):
            test_inputs[j, i-train_len, 0] = (price[i-j, column_index['close']] - price[i-j, column_index['pre_close']])/price[i-j, column_index['pre_close']]
            test_inputs[j, i-train_len, 1] = price[i-j, column_index['volume']] / price[i-j-1, column_index['volume']]
            test_inputs[j, i-train_len, 2] = price[i-j, column_index['money']] / price[i-j-1, column_index['money']]
            test_inputs[j, i-train_len, 3] = (price[i-j, column_index['close']] - price[i-j, column_index['open']]) / price[i-j, column_index['open']]
            test_inputs[j, i-train_len, 4] = (price[i-j, column_index['high']] - price[i-j, column_index['low']]) / price[i-j, column_index['low']]
            test_inputs[j, i-train_len, 5] = price[i-j, column_index['avg']] / price[i-j-1, column_index['avg']]
            test_inputs[j, i-train_len, 6] = price[i-j, column_index['open']] / price[i-j, column_index['pre_close']]
    test_labels[i-train_len] = (price[i+days_after, column_index['close']] - price[i, column_index['close']]) / price[i, column_index['close']]
# 归一化
for i in range(feature_size):
    test_inputs[:, :, i] -= feature_base[i]
    test_inputs[:, :, i] /= feature_norm[i]
test_labels -= label_base
test_labels /= label_norm
test_inputs = torch.tensor(test_inputs, dtype=torch.float32).to(device)
test_labels = torch.tensor(test_labels, dtype=torch.float32).to(device)
# 测试模型
outputs = model(test_inputs)
loss = criterion(outputs, test_labels)
print(f'Test Loss: {loss.item()}')
