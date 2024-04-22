# Note

## 聚宽数据

关于JQData的使用方式，可以参见[JQData使用指南](https://www.joinquant.com/help/api/doc?name=JQDatadoc)

## util

这一部分中编写了一些工具函数

## Torch

### LSTM模型（Long Short Term Model）

对于LSTM模型，输入参数为3-order tensor，tensor.shape为(sequence_length, batch_size, feature_size)，其中batch_size为批量大小，即有多少个个序列。batch_size为一个序列的长度，也就是time_step，而feature_size为特征维度。

## 建模

LSTM建模,以两天后的涨跌幅作为标签训练失败,猜测在这种模型中,由于选定的股票华能水电长期上涨,为了使方差损失最小,LSTM模型趋于保守,在无法每次精确预测的情况下,趋向于总是猜测略高于0的增长率使方差最小。

目前发现的是，就算在预测序列中直接告诉LSTM模型要预测的值，LSTM模型的表现和之前一样。

