---
layout: post
title: "Logistic回归"
date: 2019-11-4
categories: 机器学习
tags: 有标签分类
author: xueyaiii
---
{:toc}
<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

# 实现Logistic回归
## 原理
### 概述
- 有标签分类算法
- 主要思想：由现有数据对分类边界线进行分类  

### 数学背景
#### 回归
用一条直线对已知的数据点进行拟合
#### Sigmoid函数
- 公式：
$$\sigma(z)=\frac{1}{1+e^{-z}}$$
- $z$为0时，$\sigma(z)$为0.5，$z$为$+\infty$，$\sigma(z)$为1，$z$为$-\infty$，$\sigma(z)$为0
- 如果横坐标刻度足够大，Sigmoid函数就像一个阶跃函数，可以达到分类的目的  

#### 引入Logistic分类器
- 在每个特征$x_{i}$上乘上一个回归系数$w_{i}$,相加得到$z$(**拟合**)，将$z$代入Sigmoid函数，任何大于 0.5 的数据被分入 1 类，小于 0.5 即被归入 0 类
$$z=w_{0} x_{0}+w_{1} x_{1}+w_{2} x_{2}+\ldots+w_{n} x_{n}$$
即：
$$z=w^{T} x$$
- 我们现在的目标是求解这些回归系数$w_{i}$  

#### 基于梯度上升法确定回归系数（拟合）
- 设样本的类别标签为$y$，回归系数为$w$，样本矩阵为$x$，误差为$e$，步长为$alpha$。
- 我们的目标是最小化误差$e^{T} e$(因为是列向量)反过来就是最大化$-e^{T} e$，为消去因子，此处最大化$-\frac{1}{2} e^{T} e$
- $-\frac{1}{2} e^{T} e=-\frac{1}{2}(x w-y)^{T}(x w-y)=f(w)$拆开来就是：$f(w)=-\frac{1}{2}\left(w^{T} x^{T}-y\right)(x w-y)=-\frac{1}{2}\left(w^{T} x^{T} x w-w^{T} x^{T} y-y^{T} x w+y^{T} y\right)$
- 到这就可以用梯度上升算法，对求导可以得出$\frac{\partial f(w)}{\partial w}=x^{T} y-x^{T} x w=x^{T}(y-w x)=x^{T} e$
- 更新回归系数的公式就是：$w=w+\alpha x^{T} e$  

### Logistic 算法过程
#### Logistic 回归算法
```
初始化所有回归系数  
重复R次：  
    计算整个数据集梯度
    使用 步长乘以梯度 更新回归系数
返回回归系数
```
#### Logistic 回归 开发流程
收集数据
准备数据: 由于需要进行距离计算，因此要求数据类型为数值型。
分析数据
训练算法: 找到最佳的分类回归系数。
测试算法
## 实例
### 算法
```python
import numpy as np
import matplotlib.pyplot as plt
#解析数据
def loadDataSet(file_name):
    # dataMat为原始数据， labelMat为原始数据的标签
    dataMat = []
    labelMat = []
    
    fr = open(file_name)
    for line in fr.readlines():
        lineArr = line.strip().split()
        if len(lineArr) == 1:
            continue    # 这里如果就一个空的元素，则跳过本次循环
        # 为了方便计算，我们将 X0 的值设为 1.0 ，也就是在每一行的开头添加一个 1.0 作为 X0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

#sigmoid跳跃函数
def sigmoid(inX):
    # return 1.0 / (1 + exp(-inX))

    # Tanh是Sigmoid的变形，与 sigmoid 不同的是，tanh 是0均值的。因此，实际应用中，tanh 会比 sigmoid 更好。
    return 2 * 1.0/(1+np.exp(-2*inX)) - 1

#随机梯度下降算法（随机化）
def stocGradAscent(dataMatrix, classLabels, numIter=150):
    '''
    Desc:
        改进版的随机梯度下降，使用随机的一个样本来更新回归系数
    Args:
        dataMatrix -- 输入数据的数据特征（除去最后一列数据）
        classLabels -- 输入数据的类别标签（最后一列数据）
        numIter=150 --  迭代次数
    Returns:
        weights -- 得到的最佳回归系数
    '''
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)  # 创建与列数相同的矩阵的系数矩阵，所有的元素都是1
    # 随机梯度, 循环150,观察是否收敛
    for j in range(numIter):
        # [0, 1, 2 .. m-1]
        dataIndex = list(range(m))
        for i in range(m):
            # i和j的不断增大，导致alpha的值不断减少，但是不为0
            alpha = 4 / (
                1.0 + j + i
            ) + 0.0001  # alpha 会随着迭代不断减小，但永远不会减小到0，因为后边还有一个常数项0.0001
            # 随机产生一个 0～len()之间的一个值
            # random.uniform(x, y) 方法将随机生成下一个实数，它在[x,y]范围内,x是这个范围内的最小值，y是这个范围内的最大值。
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]] * weights))
            error = classLabels[dataIndex[randIndex]] - h
            # print weights, '__h=%s' % h, '__'*20, alpha, '__'*20, error, '__'*20, dataMatrix[randIndex]
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
            del (dataIndex[randIndex])
    return weights

#可视化展示
def plotBestFit(dataArr, labelMat, weights):
    '''
        Desc:
            将我们得到的数据可视化展示出来
        Args:
            dataArr:样本数据的特征
            labelMat:样本数据的类别标签，即目标变量
            weights:回归系数
        Returns:
            None
    '''

    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    """
    y的由来
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    w0*x0+w1*x1+w2*x2=f(x)
    x0最开始就设置为1叻， x2就是我们画图的y值，而f(x)被我们磨合误差给算到w0,w1,w2身上去了
    所以： w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
    """
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def simpleTest():
    # 1.收集并准备数据
    dataMat, labelMat = loadDataSet("TestSet.txt")

    # 2.训练模型，  f(x)=a1*x1+b2*x2+..+nn*xn中 (a1,b2, .., nn).T的矩阵值
    # 因为数组没有是复制n份， array的乘法就是乘法
    dataArr = np.array(dataMat)
    weights = stocGradAscent(dataArr, labelMat)
    print(weights)
    # 数据可视化
    plotBestFit(dataArr, labelMat, weights)

if __name__ == "__main__":
    simpleTest()
```
### 运行环境
Windows10+Anaconda Spyder（Python 3.5)
### 运行结果
- ![运行结果](/image/Logistic.PNG)
