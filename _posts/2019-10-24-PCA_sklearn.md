---
layout: post
title: "PCA主成分分析"
date: 2019-10-24
categories: 机器学习
tags: 数据预处理 数据降维
author: xueyaiii
---
{:toc}

## 用sklearn实现PCA(主成分分析)
### PCA
- 无监督数据降维技术
- 一种**特征抽取**算法

#### 特征抽取与特征选择
- 目的均为减少特征数据集的属性（特征）的数目
- 特征选择：去掉无关特征，保留相关特征，未改变原来特征空间
- 特征抽取：将机器学习算法不能识别的原始数据转化为算法可以识别的特征的过程，改变了原来的特征空间

#### PCA工作原理
1. 找出第一个主成分的方向，也就是数据**方差最大**的方向。
2. 找出第二个主成分的方向，也就是数据**方差次大**的方向，并且该方向与第一个主成分方向正交(orthogonal 如果是二维空间就叫垂直)
3. 通过这种方式计算出所有的主成分方向。
4. 通过数据集的协方差矩阵及其特征值分析，可以得到这些主成分的值。
5. 一旦得到了协方差矩阵的特征值和特征向量，我们就可以保留最大的 N 个特征。这些特征向量也给出了 N 个最重要特征的真实结构，我们就可以通过将数据乘上这 N 个特征向量 从而将它转换到新的空间上。

#### PCA算法流程
- 对原始$d$维数据集做标准化处理。
- 构造样本的协方差矩阵。(后面进行对角化)
- 计算协方差矩阵的特征值和相应的特征向量。
- 选择与前$k$个最大特征值对应的特征向量，其中$k$为新特征空间的维度$（\mathrm{k} \leq \mathrm{d})$。
- 通过前$k$个特征向量构建映射矩阵$W$
- 通过映射矩阵$W$将$d$维的输入数据集$X$转换到新的$k$维特征子空间。

#### 实现PCA
**数据预处理**  

特征标准化  

$x_{j}^{(i)}=\frac{x_{j}^{(i)}-\mu_{j}}{s_{j}}$，$\mu_{j}$为特征$j$的均值，$s_{j}$为特征$j$的标准差
```python
#数据预处理
    #加载数据集
import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)
    #将数据集分成训练集和测试集
from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, 
                     stratify=y,
                     random_state=0)
    #使用单位方差标准化数据集
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
```
**构造协方差矩阵 获得协方差矩阵的特征值和特征向量**  

协方差矩阵的特征向量代表主成分（最大方差方向），而对应的特征值大小就决定了特征向量的重要性  

计算协方差  

$\sigma_{j k}=\frac{1}{n} \sum_{i=1}^{n}\left(x_{j}^{(i)}-\mu_{j}\right)\left(x_{k}^{(i)}-\mu_{k}\right)$，$\mu_{j}$和$\mu_{k}$分别为特征$j$和$k$的均值  

协方差矩阵  

$V=\left(\begin{array}{cccc}{\sigma_{11}} & {\sigma_{12}} & {\cdots} & {\sigma_{1 n}} \\ {\sigma_{21}} & {\sigma_{22}} & {\cdots} & {\sigma_{2 n}} \\ {\vdots} & {\vdots} & {} & {\vdots} \\ {\sigma_{n 1}} & {\sigma_{n 2}} & {\cdots} & {\sigma_{n n}}\end{array}\right)$

```python
#构造协方差矩阵 获得协方差矩阵的特征值和特征向量
import numpy as np
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigenvalues \n%s' % eigen_vals)
```
**选择与前$k$个最大特征值对应的特征向量**  

绘制方差贡献率图像
```python
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

import matplotlib.pyplot as plt


plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('images/05_02.png', dpi=300)
plt.show()
```
**特征值降序排列**  

```python
#按降序排列特征值
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
#选两个对应的特征值最大的特征向量
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)
```
**通过前$k$个特征向量构建映射矩阵$W$**  

```python
X_train_std[0].dot(w)
```
**通过映射矩阵$W$将$d$维的输入数据集$X$转换到新的$k$维特征子空间**  

```python
X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_03.png', dpi=300)
plt.show()
```

### 用sklearn实现PCA
#### 步骤
- 对数据进行预处理
    - 加载数据集（使用自带葡萄酒数据）
    - 将数据集分成训练集和测试集
    - 使用单位方差标准化数据集
    - 使用PCA进行特征抽取
        本例将训练数据转换到两个主成分轴生成的决策区域
- 逻辑斯蒂回归对数据进行分类
- 对测试数据进行预测
- 使用plot_decision_region进行可视化展示
#### 运行环境
Windows10 + anaconda3 Spyder + python3
#### 运行结果
- 训练集
- ![训练集](/image/PCA_sklearn1.PNG)

- 测试集

- ![测试集](/image/PCA_sklearn2.PNG)