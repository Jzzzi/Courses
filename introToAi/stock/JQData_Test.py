import jqdatasdk as jq

# 登陆授权
jq.auth('18973738468','040725Liu')

# 查询剩余次数
print(jq.get_query_count())

# 查询账号信息
info = jq.get_account_info()
print(info)

# 获取标的
securities = jq.get_all_securities(types = ['index'], date = None)
print(securities)
print('end')

# 获取价格
price = jq.get_price('000001.XSHE', start_date = '2024-02-01', end_date = '2024-03-01')
print(price)