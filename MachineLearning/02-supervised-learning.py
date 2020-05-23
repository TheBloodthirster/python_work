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
# 然后像前面一样构建线性回归模型：
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

# 训练集和测试集之间的性能差异是过拟合的明显标志
# 因此我们应该试图找到一个可以控制复杂度的模型标准
# 线性回归最常用的替代方法之一就是岭回归（ridge regression）


# 3. 岭回归 岭回归也是一种用于回归的线性模型，因此它的预测公式与普通最小二乘法相同。
# 但在岭回归中，对系数（w）的选择不仅要在训练数据上得到好的预测结果，而且还要拟合附加约束。
# 我们还希望系数尽量小。换句话说，w 的所有元素都应接近于 0。
# 直观上来看，这意味着每个特征对输出的影响应尽可能小（即斜率很小） ，同时仍给出很好的预测结果。
# 这种约束是所谓正则化（regularization）的一个例子。
# 正则化是指对模型做显式约束，以避免过拟合。岭回归用到的这种被称为 L2 正则化

from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train,y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test))) 

# 可以看出，Ridge 在训练集上的分数要低于 LinearRegression，但在测试集上的分数更高。 
# 这和我们的预期一致。线性回归对数据存在过拟合。Ridge 是一种约束更强的模型，所以更不容易过拟合。
# 复杂度更小的模型意味着在训练集上的性能更差，但泛化性能更好。
# 由于我们只对泛化性能感兴趣，所以应该选择 Ridge 模型而不是 LinearRegression 模型。 


# Ridge 模型在模型的简单性（系数都接近于 0）与训练集性能之间做出权衡。
# 简单性和训练 集性能二者对于模型的重要程度可以由用户通过设置 alpha 参数来指定。

# 增大 alpha 会使得系数更加趋向于 0，从而降低训练集性能， 但可能会提高泛化性能。
ridge10 = Ridge(alpha=10).fit(X_train,y_train)
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test))) 

#减小 alpha 可以让系数受到的限制更小，即在图 2-1 中向右移动。
# 对于非常小的 alpha 值， 系数几乎没有受到限制，我们得到一个与 LinearRegression 类似的模型
ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)
print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test))) 

plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend() 

# 更大的 alpha 表示约束更强的模型，所以我们预计大 alpha 对应的 coef_ 元素比小 alpha 对应的 coef_ 元素要小。

# 4. lasso 除了 Ridge，还有一种正则化的线性回归是 Lasso。
# 与岭回归相同，使用 lasso 也是约束系 数使其接近于 0，但用到的方法不同，
# 叫作 L1 正则化。 8L1 正则化的结果是，使用 lasso 时 某些系数刚好为 0。
# 这说明某些特征被模型完全忽略。这可以看作是一种自动化的特征选择。
# 某些系数刚好为 0，这样模型更容易解释，也可以呈现模型最重要的特征。 

from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train,y_train) 
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

# 如你所见，Lasso 在训练集与测试集上的表现都很差。
# 这表示存在欠拟合，我们发现模型 只用到了 105 个特征中的 4 个。
# 与 Ridge 类似，Lasso 也有一个正则化参数 alpha，可以控制系数趋向于 0 的强度。
# 在上一个例子中，我们用的是默认值 alpha=1.0。为了降低欠拟 合，我们尝试减小 alpha。
# 这么做的同时，我们还需要增加 max_iter 的值（运行迭代的最大次数）
lasso001 = Lasso(alpha=0.01,max_iter=100000).fit(X_train,y_train) 
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))


# alpha 值变小，我们可以拟合一个更复杂的模型，在训练集和测试集上的表现也更好。
# 模型性能比使用 Ridge 时略好一点，而且我们只用到了 105 个特征中的 33 个。
# 这样模型可能更容易理解。 但如果把alpha 设得太小，那么就会消除正则化的效果，并出现过拟合，
# 得到与 LinearRegression 类似的结果：
lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0))) 


plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
# %%
