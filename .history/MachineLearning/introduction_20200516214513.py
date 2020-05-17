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



