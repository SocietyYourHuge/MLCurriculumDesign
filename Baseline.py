import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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

# 随机森林
clf = RandomForestClassifier(n_estimators=1500, criterion='gini', min_samples_split=3,
                             min_samples_leaf=1, bootstrap=True, n_jobs=2)

# 进行结果预测
clf.fit(train_data, train_labels)
out = clf.predict(test_data)
submit = pd.read_csv("sample_submission.csv")
submit['CLASS'] = out
submit.to_csv('submission.csv', index=None)
