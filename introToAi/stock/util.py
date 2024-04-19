# Description: 一些工具函数

# 定义异常
class MyException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

# 根据名称获取标的代码
def get_security (name = None, types = ['index', 'stock']):
    import collections.abc
    from jqdatasdk import get_all_securities
    # 如果名称为空，抛出异常
    if name == None:
        raise MyException('name is None')
    # 如果名称是一个列表，那么遍历列表，获取每个标的的代码
    if isinstance(name, collections.abc.Iterable) and not isinstance(name, str):
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
    # 如果名称是一个字符串，那么直接获取标的代码
    securities = get_all_securities(types = types, date = None)
    target = securities[securities['display_name'] == name]
    if target.empty:
        raise MyException('No such security')
    code = target.index[0]
    return code

# 获取价格
def get_price_to_csv (code = None, start_date = None, end_date = None):
    from jqdatasdk import get_price
    import pandas as pd
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
    price = get_price(code, start_date = start_date, end_date = end_date)
    # 导出价格
    price.to_csv(f'{code}.csv')
    print(f'{code}.csv has been saved')
    return