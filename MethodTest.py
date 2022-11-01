import numpy as np
import pandas as pd

# 数据相对路径
train_file = "train.csv"
test_file = "test.csv"


# 分离数据集
def data_precess(filename):
    df = pd.read_csv(filename)
    data = []
    for i in df:
        # 数据转换为32位浮点数
        data.append(df[i].astype(np.float32))
    # 如果是训练集
    if 'CLASS' in df:
        # 获取类别标签，数据转换为长整型
        label = np.array(df['CLASS'].astype(np.compat.long))
        # 去掉前面的序号和后面的标签
        data = np.array(data)[1:-1].T
    # 如果是测试集
    else:
        label = np.array([])
        # 去掉前面的序号
        data = np.array(data)[1:].T
    # 返回数据和标签
    return data, label


# 获取数据
train_data, train_labels = data_precess(train_file)
test_data, _ = data_precess(test_file)
# 进行数据归一化
train_data = (train_data + 1) / 2
test_data = (test_data + 1) / 2

'''
1、SVC支持向量机
'''
from sklearn.model_selection import cross_val_score
from sklearn import svm

clf = svm.SVC(C=1, kernel='poly', degree=2, gamma='scale', coef0=0.0, probability=True)
scores = cross_val_score(clf, train_data, train_labels, cv=10)
print("1、SVC")
print(scores)
print("结果精确度：\t %0.2f\n标准差：\t\t %0.2f" % (scores.mean(), scores.std()))

'''
2、决策树
'''
from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=None, min_samples_split=7,
                                  min_samples_leaf=1)
scores = cross_val_score(clf, train_data, train_labels, cv=10)
print("2、决策树")
print(scores)
print("结果精确度：\t %0.2f\n标准差：\t\t %0.2f" % (scores.mean(), scores.std()))

'''
3、随机森林
'''
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=1500, criterion='gini', min_samples_split=3,
                             min_samples_leaf=1, bootstrap=True, n_jobs=2)
scores = cross_val_score(clf, train_data, train_labels, cv=10)
print("3、随机森林")
print(scores)
print("结果精确度：\t %0.2f\n标准差：\t\t %0.2f" % (scores.mean(), scores.std()))

'''
4、AdaBoost
'''
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(
    base_estimator=svm.SVC(C=1, kernel='poly', degree=2, gamma='scale', coef0=0.0, probability=True),
    n_estimators=100, learning_rate=0.7, random_state=None)
scores = cross_val_score(clf, train_data, train_labels, cv=10)
print("4、AdaBoost")
print(scores)
print("结果精确度：\t %0.2f\n标准差：\t\t %0.2f" % (scores.mean(), scores.std()))

'''
5、梯度提升
'''
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.7, subsample=1.0, loss='log_loss',
                                 criterion='friedman_mse')
scores = cross_val_score(clf, train_data, train_labels, cv=10)
print("5、梯度提升")
print(scores)
print("结果精确度：\t %0.2f\n标准差：\t\t %0.2f" % (scores.mean(), scores.std()))
