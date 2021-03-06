# import numpy as np
# x = np.array([[1, 2, 3], [4, 5, 6]])
# print("x:\n{}".format(x))

# from scipy import sparse

# eye = np.eye(4)
# print("NumPy array:\n{}".format(eye))

# # 将NumPy数组转换为CSR格式的SciPy稀疏矩阵 
# # 只保存非零元素
# sparse_matrix = sparse.csr_matrix(eye)
# print("\n Scipy sparse CSR matrix:\n{}".format(sparse_matrix))

# #COO格式
# data = np.ones(4)
# row_indices = np.arange(4)
# col_indices = np.arange(4)
# eye_coo = sparse.coo_matrix((data,(row_indices,col_indices)))
# print("\nCOO representation:\n{}".format(eye_coo)) 


# #%%
# import numpy as np
# import matplotlib.pyplot as plt

# # 在-10和10之间生成一个数列，共100个数
# x = np.linspace(-10, 10, 100) 
# # 用正弦函数创建第二个数组 
# y = np.sin(x) 
# # plot函数绘制一个数组关于另一个数组的折线图 
# plt.plot(x, y, marker="x") 

#%%
# import pandas as pd
# from IPython.display import display

# data = {'Name': ["John", "Anna", "Peter", "Linda"], 
#         'Location' : ["New York", "Paris", "Berlin", "London"], 
#         'Age' : [24, 13, 53, 33] }

# data_pandas = pd.DataFrame(data)
# display(data_pandas[data_pandas.Age > 30])



# import sys
# print("Python version:", sys.version)

# import pandas as pd
# print("pandas version:", pd.__version__)

# import matplotlib
# print("matplotlib version:", matplotlib.__version__)

# import numpy as np
# print("NumPy version:", np.__version__)

# import scipy as sp
# print("SciPy version:", sp.__version__)

# import IPython
# print("IPython version:", IPython.__version__)

# import sklearn
# print("scikit-learn version:", sklearn.__version__)



################################################################################
#########################第一个应用：鸢尾花分类##################################
################################################################################
from sklearn.datasets import load_iris
iris_dataset = load_iris()
# print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

# print(iris_dataset['DESCR'][:1000] + "\n...") 

print("Target names: {}".format(iris_dataset['target_names'])) 
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape)) 