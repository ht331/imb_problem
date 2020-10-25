from prepare.data_split import read_bunch, write_bunch
import itertools
import numpy as np
import bunch
import pandas as pd


def covert_group(h, act_fun, data_name):

    data_set = {}
    for af in act_fun:
        data = read_bunch('../data/%s_%s_%s.data' % (data_name, af, h))
        data_set[af] = data

    # 两两分组
    pool_2 = list(itertools.combinations(act_fun, 2))
    # 三三分组
    pool_3 = list(itertools.combinations(act_fun, 3))

    pool = [pool_2, pool_3]
    for p in pool:
        for pp in p:

            daf_train = []
            daf_test = []
            y_train = []
            y_test = []
            table_name = '_'.join(pp)
            for i in range(len(pp)):
                a = pp[i]
                d = data_set[a]
                tr = d.X_train
                te = d.X_test
                y_train = d.y_train
                y_test = d.y_test
                if i == 0:
                    daf_train = tr
                    daf_test = te
                else:

                    daf_train = np.hstack((daf_train, tr))
                    daf_test = np.hstack((daf_test, te))

            daf_data = bunch.Bunch(X_train=daf_train,
                                   y_train=y_train,
                                   X_test=daf_test,
                                   y_test=y_test)
            write_bunch('e:/Paper/imb_problem/data/%s_%s_%s.data' % (data_name, table_name, h), daf_data)


if __name__ == '__main__':
    covert_group(10, ['sigmoid', 'tanh', 'relu', 'elu', 'selu', 'swish', 'leakyrelu'], 'DAF_data')

