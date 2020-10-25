import pandas as pd
import math
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler, NeighbourhoodCleaningRule, TomekLinks
from prepare.data_split import read_bunch
import itertools


def gdbt(data):
    X_train = pd.DataFrame(data.X_train)
    X_test = data.X_test
    y_train = data.y_train
    y_test = data.y_test
    osp = RandomUnderSampler(random_state=10)
    X_train, y_train = osp.fit_sample(X_train, y_train)

    # clf = HistGradientBoostingClassifier(random_state=10)
    clf = GradientBoostingClassifier(random_state=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    c_m = metrics.confusion_matrix(y_test, y_pred)

    print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
    print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
    print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
    print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
    print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
    print("F-measure:%.4f" % (metrics.recall_score(y_test, y_pred) * metrics.precision_score(y_test, y_pred)))

    tpr = c_m[1][1] / (c_m[1][1] + c_m[1][0])
    tnr = c_m[0][0] / (c_m[0][0] + c_m[0][1])
    g_mean = math.sqrt(tpr * tnr)
    print('G-mean: %.4f' % g_mean)
    return metrics.roc_auc_score(y_test, y_pred)


if __name__ == '__main__':
    from prepare.compare import covert_group
    roc_list = []
    print('control group')
    data = read_bunch('../data/creditcard.data')
    ras = gdbt(data)
    print('=' * 20)
    roc_list.append(['control group', ras])

    print('experience group')
    h = 10
    data_name = 'creditcard'
    act_func = ['sigmoid', 'tanh', 'elu', 'selu']
    covert_group(h, act_func, data_name)

    pool_2 = list(itertools.combinations(act_func, 2))
    # 三三分组
    pool_3 = list(itertools.combinations(act_func, 3))
    pool = pool_2 + pool_3
    for p in pool:
        act_func.append('_'.join(p))

    for af in act_func:
        print(af)
        data = read_bunch('../data/%s_%s_%s.data' % (data_name, af, h))
        ras = gdbt(data)
        print('-' * 20)
        roc_list.append([af, ras])

    roc_df = pd.DataFrame(data=roc_list)
    roc_df.to_excel('../feature/compare_result/hidden_%s.xlsx' % h)

