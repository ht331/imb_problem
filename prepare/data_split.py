"""
    使用清洗后的数据进行预处理
    1、使用均值插补；
    2、规范化；
    3、训练集、测试集按照 7:3 的比例划分好
    4、存放为 data_set.data
"""

import pandas as pd
import bunch
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def read_bunch(path):
    file_obj = open(path, 'rb')
    bunch_ = pickle.load(file_obj)
    file_obj.close()
    return bunch_


def write_bunch(path, bunch_obj):
    file_obj = open(path, 'wb')
    pickle.dump(bunch_obj, file_obj)
    file_obj.close()


def scale_split():

    data = pd.read_excel('../data/data_clean.xlsx')
    y = data['逾期标签']
    x = data[[i for i in list(data.columns) if i != '逾期标签']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                        random_state=10)
    imp = SimpleImputer(strategy='mean')
    x1 = imp.fit_transform(x_train)
    x0 = imp.transform(x_test)

    prep = StandardScaler()
    x1 = prep.fit_transform(x1)
    x0 = prep.transform(x0)

    dataset = bunch.Bunch(X_train=x1,
                          y_train=y_train,
                          X_test=x0,
                          y_test=y_test)
    write_bunch('../data/data_set.data', dataset)


def convert_data():

    data = pd.read_csv('../data/creditcard.csv', low_memory=False)
    x = data.iloc[:, 1:-2]
    y = data['Class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                        random_state=10)
    prep = StandardScaler()
    x_train = prep.fit_transform(x_train)
    x_test = prep.transform(x_test)
    dataset = bunch.Bunch(X_train=x_train,
                          y_train=y_train,
                          X_test=x_test,
                          y_test=y_test)
    write_bunch('../data/creditcard.data', dataset)


if __name__ == '__main__':
    convert_data()