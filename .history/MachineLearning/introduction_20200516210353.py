import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))

from scipy import sparse

eye = np.eye(4)
print("NumPy array:\n{}".format(eye))

# 将NumPy数组转换为CSR格式的SciPy稀疏矩阵 
# # 只保存非零元素
