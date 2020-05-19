#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn

# 生成数据集
X, y = mglearn.datasets.make_forge()

# 数据及绘图
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.legend(["class 0","class 1"],loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X shape:{}".format(X.shape))

mglearn.plots.plot_knn_classification(n_neighbors=5) 

X1, y1 = mglearn.datasets.make_wave()

# 数据及绘图
plt.plot(X1,y1,'o')
plt.ylim(-3,3)
plt.xlabel("Feature")
plt.ylabel("Target")
print("X shape:{}".format(X1.shape))



#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys():\n{}".format(cancer.keys()))
print("Shape of cancer data: {}".format(cancer.data.shape)) 
print("Sample cpimts per class:\n{}".format(
    {n: v for n,v in zip(cancer.target_names,np.bincount(cancer.target))}
))
print("Feature names:\n{}".format(cancer.feature_names)) 


from sklearn.datasets import load_boston
boston = load_boston()
print("Data shape: {}".format(boston.data.shape))
X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))





################################################################################
# k-Nearest Neighbors
# k-Neighbors classification
# 现在看一下如何通过 scikit-learn 来应用 k 近邻算法。
################################################################################
#%%
# 首先，正如第 1 章所述，将数据分 为训练集和测试集，以便评估泛化性能：
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X,y = mglearn.datasets.make_forge()

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)

#然后，导入类并将其实例化。这时可以设定参数，比如邻居的个数。这里我们将其设为 3：

clf = KNeighborsClassifier(n_neighbors=1)

# 现在，利用训练集对这个分类器进行拟合。对于 KNeighborsClassifier 来说就是保存数据 集，以便在预测时计算与邻居之间的距离：
clf.fit(X_train,y_train)
print("Test set predictions: {}".format(clf.predict(X_test))) 

# 为了评估模型的泛化能力好坏，我们可以对测试数据和测试标签调用 score 方法： 
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test))) 

################################################################################
# 分析KNeighborsClassifier
# 对于二维数据集，我们还可以在 xy 平面上画出所有可能的测试点的预测结果。
# 我们根据平面中每个点所属的类别对平面进行着色。这样可以查看决策边界（decision boundary）， 
# 即算法对类别 0 和类别 1 的分界线。
################################################################################
fig, axes = plt.subplots(1, 3, figsize=(10, 3)) 
for n_neighbors,ax in zip([1,3,9],axes):
    # fit方法返回对象本身，所以我们可以将实例化和拟合放在一行代码中
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill=True,ax=ax,alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
################################################################################
# 从左图可以看出，使用单一邻居绘制的决策边界紧跟着训练数据。
# 随着邻居个数越来越多，决策边界也越来越平滑。更平滑的边界对应更简单的模型。
# 换句话说，使用更少的邻 居对应更高的模型复杂度（如图 2-1 右侧所示） ，
# 而使用更多的邻居对应更低的模型复杂度 （如图 2-1 左侧所示） 。
################################################################################






# %%
################################################################################
# 我们来研究一下能否证实之前讨论过的模型复杂度和泛化能力之间的关系。
# 我们将在现实 世界的乳腺癌数据集上进行研究。先将数据集分成训练集和测试集，
# 然后用不同的邻居个数对训练集和测试集的性能进行评估
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(
    cancer.data,cancer.target,stratify=cancer.target,random_state = 66
    )
training_accuracy = []
test_accuracy = []

# n_neighbors取值从1-10
neighbors_settings = range(1,11)
for n_neighbors in neighbors_settings:
    # 构建模型
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    # 记录训练精度
    training_accuracy.append(clf.score(X_train,y_train))
    #记录泛化精度
    test_accuracy.append(clf.score(X_test,y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

# %%
################################################################################
# 3. k近邻回归k 近邻算法还可以用于回归。
# 我们还是先从单一近邻开始，这次使用 wave 数据集。
# 我们添加了 3 个测试数据点，在 x 轴上用绿色五角星表示。
# 利用单一邻居的预测结果就是最近邻 的目标值。
################################################################################

#mglearn.plots.plot_knn_regression(n_neighbors=1) 
# 同样，也可以用多个近邻进行回归。在使用多个近邻时，预测结果为这些邻居的平均值 
#mglearn.plots.plot_knn_regression(n_neighbors=3)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)
# 将wave数据集分为训练集和测试集 
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

# 模型实例化，并将邻居个数设为3 
reg = KNeighborsRegressor(n_neighbors=3)

# 利用训练数据和训练目标值来拟合模型
reg.fit(X_train,y_train)

# test
print("Test set predictions:\n{}".format(reg.predict(X_test))) 

# 对于回归问题，这一方法返回的是 R 2 分数。
# R 2 分 数也叫作决定系数，是回归模型预测的优度度量，位于 0 到 1 之间。
# R 2 等于 1 对应完美预 测，R 2 等于 0 对应常数模型，
# 即总是预测训练集响应（y_train）的平均值
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test))) 
################################################################################
# 分析KNeighborsRegresso
################################################################################
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# 创建1000个数据点，在-3和3之间均匀分布
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors,ax in zip([1,3,9],axes):
    # 利用1个、3个或9个邻居分别进行预测 
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    ax.set_title(
        "{} neighbor(s)\n train score:{:.2f} test score:{:.2f}".format(
            n_neighbors,reg.score(X_train,y_train),
            reg.score(X_test,y_test)
        )
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target", "Test data/target"], loc="best")







































################################################################################
# 2.3.3 线性模型 线性模型利用输入特征的线性函数（linear function）进行预测
# 对于回归问题，线性模型预测的一般公式如下： 
# ŷ = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b 
# 这里 x[0] 到 x[p] 表示单个数据点的特征（本例中特征个数为 p+1）， 
# w 和 b 是学习模型的 参数，ŷ 是模型的预测结果。
################################################################################
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split

mglearn.plots.plot_linear_regression_wave() 

#  2. 线性回归（又名普通最小二乘法）
from sklearn.linear_model import LinearRegression
X,y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train,y_train)
print("lr.coef_:{}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_)) 

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test))) 

# %%
# 我们来看一下 LinearRegression 在更 复杂的数据集上的表现，比如波士顿房价数据集。
# 记住，这个数据集有 506 个样本和 105 个导出特征。首先，加载数据集并将其分为训练集和测试集。
# 然后像前面一样构建线性回 归模型：
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split

X,y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train,y_train)
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test))) 

# %%
