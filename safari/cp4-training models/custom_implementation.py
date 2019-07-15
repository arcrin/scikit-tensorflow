import numpy as np
import os
#%% change working directory
os.chdir('/Users/arcia/Project/Python/Scikit-Learn and Tensor Flow/cp4-training models')

#%%
v = np.array([[1, 2], [3, 4], [5, 6]])
v[:, 1] = [10, 11, 12]  # assign new values to column 2, np arrays are 0 indexed
v = np.c_[v, [100, 101, 102]]  # append new column to v
print(v.ravel())  # flatten with default order='C'. rows
print(v.ravel(order='F'))  # flatten with order='F'. columns

#%%
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[11, 12], [13, 14], [15, 16]])
C = np.c_[A, B]
D = np.concatenate((A, B))

#%% Operations
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[11, 12], [13, 14], [15, 16]])
C = np.array([[1, 1], [2, 2]])

print(A.dot(C))  # Matrix multiplication
print(A * B)  # element-wise multiplication, numpy default
print(A ** 2)  # square each element

v = np.array([[1], [2], [3]])
print(1 / v)
print(v + np.ones((np.shape(v)[0], 1)))
print(A.T)  # transpose
a = [1, 15, 2, 0.5]
print(np.amax(a))  # find max value of an np array
print(np.where(a == np.amax(a)))  # index of the max value
print(np.where(A == 5))  # index of max value of a matrix

A = np.array([[8, 1, 6], [3, 5, 7], [4, 9, 2]])
print(np.where(A >= 7))
print(np.sum(a))
print(np.prod(a))

print(np.amax(A, axis=0))  # max along first dimension (max of each column)
print(np.amax(A, axis=1))  # max along seconds dimension (max of each row)

print(np.sum(A, axis=0))
print(np.sum(A, axis=1))

print(np.trace(A))  # sum of diagonal
print(np.sum(A * np.flipud(np.eye(3))))  # sum of the other diagonal

print(np.linalg.inv(A).dot(A))
print(np.linalg.pinv(A).dot(A))
