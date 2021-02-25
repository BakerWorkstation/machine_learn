'''
Author: sdc
Date: 2021-01-08 14:40:14
LastEditTime: 2021-01-14 16:18:42
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /opt/sample/3.py
'''

# import evtx
import sklearn
import numpy as np
import pandas as pd
from sklearn import metrics
from time import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

import IPython
print("IPython version: {}".format(IPython.__version__))
import platform
print("platform  version: {}".format(platform .__version__))
print("python version: {}".format(platform.python_version()))
print("numpy version: {}".format(np.__version__))
print("pandas version: {}".format(pd.__version__))
print("scikit-learn version: {}".format(sklearn.__version__))


# 定义读取、处理数据集函数
def data_processing(file, all_features=True):
    fr = pd.read_csv(file, encoding='utf-8', error_bad_lines=False, nrows=None)
    data = np.array(fr)
    print('数据集大小：',data.shape)

    data[:,-1] = LabelEncoder().fit_transform(data[:,-1])        # 标签的编码
    data[:,0:-1] = OrdinalEncoder().fit_transform(data[:,0:-1])  # 特征的分类编码
    data = StandardScaler().fit_transform(data)                  # 标准化：利用Sklearn库的StandardScaler对数据标准化

    # 选取特征和标签
    line_nums = len(data)
    data_label = np.zeros(line_nums)
    if all_features == True:
        data_feature = np.zeros((line_nums, 41))   # 创建line_nums行 41列的矩阵
        for i in range(line_nums):                 # 依次读取每行
            data_feature[i,:] = data[i][0:41]      # 选择前41个特征  划分数据集特征和标签
            data_label[i]  = int(data[i][-1])      # 标签
    else:
        data_feature = np.zeros((line_nums, 10))   # 创建line_nums行 10列的矩阵
        for i in range(line_nums):                 # 依次读取每行
            feature = [3,4,5,6,8,10,13,23,24,37]   # 选择第3,4,5,6,8,10,13,23,24,37这10个特征分类
            # (3) service - 目标主机的网络服务类型，离散类型
            # (4) flag - 连接正常或错误的状态，离散类型
            # (5) src_bytes - 从源主机到目标主机的数据的字节数，连续类型
            # (6) dst_bytes - 从目标主机到源主机的数据的字节数，连续类型
            # (8) wrong_fragment - 错误分段的数量，连续类型
            # (10) hot - 访问系统敏感文件和目录的次数，连续
            # (13) num_compromised - compromised条件出现的次数，连续
            # (23) count - 过去两秒内，与当前连接具有相同的目标主机的连接数，连续
            # (24) srv_count - 过去两秒内，与当前连接具有相同服务的连接数，连续
            # (37) dst_host_srv_diff_host_rate - 前100个连接中，与当前连接具有
            #     相同目标主机相同服务的连接中，与当前连接具有不同源主机的连接所占的百分比，连续
            for j in feature:
                data_feature[i,feature.index(j)] = data[i][j]
            data_label[i]  = int(data[i][-1])      # 标签

    print('数据集特征大小：',data_feature.shape)
    print('数据集标签大小：',len(data_label))
    return data_feature, data_label

data_feature, data_label = data_processing(file="kddcup.data.one.txt", all_features=False)
# test_feature, test_label = data_processing(file="kdd.correct.one.txt", all_features=False)
train_feature, test_feature, train_label, test_label = train_test_split(data_feature, data_label,test_size=0.5,random_state=5)# 测试集40%
print('训练集特征大小：{}，训练集标签大小：{}'.format(train_feature.shape, train_label.shape))
print('测试集特征大小：{}，测试集标签大小：{}'.format(test_feature.shape, test_label.shape))

# 决策树DT
if __name__ == '__main__':
    begin_time = time()                     # 训练预测开始时间
    print('Start training DT：',end='')     # CART
    dt = sklearn.tree.DecisionTreeClassifier(criterion='gini',splitter='best', max_depth=20, min_samples_split=2, min_samples_leaf =1)
    dt.fit(train_feature, train_label)
    print(dt)
    print('Training done！')
    print('Start prediction DT：')
    test_predict = dt.predict(test_feature)
    print('Prediction done！')
    print('预测结果：',test_predict)
    print('实际结果：',test_label)
    print('正确预测的数量：',sum(test_predict==test_label)) 
    print('准确率:', metrics.accuracy_score(test_label, test_predict))                         # 预测准确率输出
    print('宏平均精确率:',metrics.precision_score(test_label,test_predict,average='macro'))    # 预测宏平均精确率输出
    print('微平均精确率:', metrics.precision_score(test_label, test_predict, average='micro')) # 预测微平均精确率输出
    print('宏平均召回率:',metrics.recall_score(test_label,test_predict,average='macro'))       # 预测宏平均召回率输出
    print('平均F1-score:',metrics.f1_score(test_label,test_predict,average='weighted'))        # 预测平均f1-score输出
    end_time = time()                        # 训练预测结束时间
    total_time = end_time - begin_time
    print('训练预测耗时：',total_time,'s')


    print('混淆矩阵输出:')
    print(metrics.confusion_matrix(test_label,test_predict))                                   # 混淆矩阵输出
    # 从精确率:precision、召回率:recall、 调和平均f1值:f1-score和支持度:support四个维度进行衡量
    print('分类报告:')
    print(metrics.classification_report(test_label, test_predict))     