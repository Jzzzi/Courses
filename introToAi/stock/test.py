from util import *
from jqdatasdk import *
auth('18973738468','040725Liu')
# code = get_security(['深证成指','上证指数'])
# price = get_price(code, start_date = '2024-02-01', end_date = '2024-03-01')
# print(type(price))
# print(price.info())
# print(price)
get_price_to_csv('000001.XSHE', start_date = '2024-02-01', end_date = '2024-03-01')