
from numpy.lib.arraysetops import ediff1d
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('ex8data1.xlsx', sheet_name='X', header=None)
df.head()


# 检查此数据集中有多少个训练示例
m = len(df)
# 计算每个特征的平均值。这里我们只有两个特征：0和1。
s = np.sum(df, axis=0)
mu = s/m
print('每个特征的平均值:\n' , mu)

# 计算方差
vr = np.sum((df - mu)**2, axis=0)
variance = vr/m
print('每个特征的方差:\n' , variance)

# 做成对角线形状
var_dia = np.diag(variance)
print('做成对角线形状:\n' , var_dia)

# 计算概率
k = len(mu)
X = df - mu
p = 1/((2*np.pi)**(k/2)*(np.linalg.det(var_dia)**0.5))* np.exp(-0.5* np.sum(X @ np.linalg.pinv(var_dia) * X, axis=1))
print('计算概率:\n' , p)
print(p.describe())

# 找出阈值概率  交叉验证数据和标签
cvx = pd.read_excel('ex8data1.xlsx', sheet_name='Xval', header=None)
cvx.head()

cvy = pd.read_excel('ex8data1.xlsx', sheet_name='y', header=None)
cvy.head()

# 把'cvy'转换成NumPy数组
y = np.array(cvy)

# 定义一个函数来计算真正例、假正例和假反例
def tpfpfn(ep):
    tp, fp, fn = 0, 0, 0
    for i in range(len(y)):
        if p[i] <= ep and y[i][0] == 0:
            tp += 1
        elif p[i] <= ep and y[i][0] == 0:
            fp += 1
        elif p[i] > ep and y[i][0] == 1:
            fn += 1
    return tp, fp, fn

# 列出低于或等于平均概率的概率
eps = [i for i in p if i <= p.mean()]

# 定义一个计算f1分数的函数
def f1(ep):
    tp, fp, fn = tpfpfn(ep)
    prec = tp/(tp + fp)
    rec = tp/(tp + fn)
    f1 = 2*prec*rec/(prec + rec)
    return f1

# 计算所有epsilon或我们之前选择的概率值范围的f1分数
# f分数通常在0到1之间，其中f1得分越高越好
f = []
for i in eps:
    f.append(f1(i))
print('f分数:\n' , f)

print(max(f))
# 使用“argmax”函数来确定f分数值最大值的索引
index = np.array(f).argmax()

# 现在用这个索引来得到阈值概率
e = eps[index]
print('阈值概率:\n' , e)

label = []
for i in range(len(df)):
    if p[i] <= e:
        label.append(1)
    else:
        label.append(0)
print(label)

df['label'] = np.array(label)
df.head()


plt.figure()
plt.scatter(df[0], df[1])
plt.show()