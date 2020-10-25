import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler, NeighbourhoodCleaningRule, TomekLinks
from prepare.data_split import read_bunch

data = read_bunch('../data/data_set.data')
X_train = pd.DataFrame(data.X_train)
X_test = data.X_test
y_train = data.y_train
y_test = data.y_test

# osp = SMOTE(random_state=10)
osp = RandomUnderSampler(random_state=10)
X_train, y_train = osp.fit_sample(X_train, y_train)  # SMOTE

clf = RandomForestClassifier(n_estimators=100, random_state=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# y_pred = my_adaboost_clf(y_train, X_train, y_test, X_test, M=100)

c_m = metrics.confusion_matrix(y_test, y_pred)
print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
print("F-measure:%.4f" % (metrics.recall_score(y_test, y_pred) * metrics.precision_score(y_test, y_pred)))