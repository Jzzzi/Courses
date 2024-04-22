# Description: 一些工具函数

class MyException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class DebugStop(Exception):
    def __init__(self):
        self.message = 'debug stop'
        super().__init__(self.message)

def get_security (name = None, types = ['index', 'stock']):
    '''
    根据名称获取标的代码
    当为单个名称时，返回一个字符串
    当为多个名称时，返回一个列表
    '''
    import collections.abc
    from jqdatasdk import get_all_securities, auth
    # 登陆授权
    auth('18973738468','040725Liu')
    # 如果名称为空，抛出异常
    if name == None:
        raise MyException('name is None')
    # 如果名称是一个字符串，那么直接获取标的代码
    if isinstance(name, str):
        securities = get_all_securities(types = types, date = None)
        target = securities[securities['display_name'] == name]
        if target.empty:
            raise MyException('No such security')
        code = target.index[0]
        return code
    # 如果名称是一个列表，那么遍历列表，获取每个标的的代码
    elif isinstance(name, collections.abc.Iterable) and not isinstance(name, str):
        names = name
        codes = []
        securities = get_all_securities(types = types, date = None)
        for name in names:
            target = securities[securities['display_name'] == name]
            if target.empty:
                raise MyException('No such security')
            code = target.index[0]
            codes.append(code)
        return codes
    # 如果名称不是字符串，也不是列表，抛出异常
    else:
        raise MyException('name is not a string or a list')

def get_price_to_csv (code = None, start_date = None, end_date = None,
                      fields = ['open', 'close', 'high', 'low', 'volume', 'money', 'pre_close', 'avg']):
    '''
    获取价格并保存到csv文件
    '''
    from jqdatasdk import get_price, auth
    import pandas as pd
    # 登陆授权
    auth('18973738468','040725Liu')
    # 如果标的代码为空，抛出异常
    if code == None:
        raise MyException('code is None')
    # 如果开始日期为空，抛出异常
    if start_date == None:
        raise MyException('start_date is None')
    # 如果结束日期为空，抛出异常
    if end_date == None:
        raise MyException('end_date is None')
    # 获取价格
    price = get_price(code, start_date = start_date, end_date = end_date, fields = fields)
    # 导出价格
    price.to_csv(f'{code}.csv')
    print(f'{code}.csv has been saved')
    return

def batchify_sequences(data, batch_size, shuffle=True):
    import torch
    """
    将序列数据集划分成不同的批次，适用于LSTM网络。

    参数:
    data -- 三维张量，形状为(num_samples, sequence_length, num_features)
    batch_size -- 整数，每个批次的样本数量
    shuffle -- 布尔值，表示是否在每次迭代前打乱数据集

    返回:
    一个生成器，每次生成一个批次的张量
    """
    if shuffle:
        indices = torch.randperm(data.size(0))  # 生成随机索引
        data = data[indices]  # 根据随机索引打乱数据

    # 生成批次
    for i in range(0, data.size(0), batch_size):
        yield data[i:i + batch_size]
