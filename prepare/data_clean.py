"""
    数据的预处理
    1、性别编码 男 1 女 0；
    2、去掉收货地址和户籍 两个文本类型数据
    3、分数为0的值换为nan值
"""


import pandas as pd
import numpy as np


def prepare():

    data = pd.read_excel('../data/data.xlsx')
    del data['ID']
    del data['收货地址']
    del data['户籍']

    data['分数'].replace(0, np.nan, inplace=True)
    data['性别'].replace(0, '女', inplace=True)
    data['性别'].replace(1, '男', inplace=True)

    data.to_excel('../data/data_clean.xlsx')


if __name__ == '__main__':
    prepare()