# sieve = [True] * 101
# for i in range(2, 100):
#     if sieve[i]:
#         print(i)
#         for j in range(i*i,100,i):
#             sieve[j]=False
import numpy as np

# a = set(['hello','big'])
# b = set(['hello','small'])
# print(a&b)



a = np.array([
    [1,1,1],
    [2,2,2],
    [3,3,3],
    [4,4,4],
    [5,5,5]
])
b = np.array([
    [1,1],
    [1,1],
    [2,2]
])
c = a.dot(b)
print(c)
def cla_original_mat(inverse_mat):
    dimension = np.shape(inverse_mat)[1]
    unit_mat = np.identity(dimension)
    original_mat = unit_mat*inverse_mat.I
    return original_mat

xv = cla_original_mat(np.mat(a))*c
print("\n")
print(xv)
print("\n")
print(a.dot(xv))