# Note

## 聚宽数据

关于JQData的使用方式，可以参见[JQData使用指南](https://www.joinquant.com/help/api/doc?name=JQDatadoc)

## util

这一部分中编写了一些工具函数

## Torch

### LSTM模型（Long Short Term Model）

对于LSTM模型，输入参数为3-order tensor，tensor.shape为(sequence_length, batch_size, feature_size)，其中batch_size为批量大小，即有多少个个序列。batch_size为一个序列的长度，也就是time_step，而feature_size为特征维度。

## naivemodel

