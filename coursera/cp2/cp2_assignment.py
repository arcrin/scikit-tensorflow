import numpy as np
import os
from matplotlib import pyplot as plt
#%%
os.chdir('/Users/arcia/Project/Python/Scikit-Learn and Tensor Flow/coursera/cp2')
#%%
A = np.eye(5)

#%%
data = np.loadtxt('ex1data1.txt', delimiter=",")
x = data[:, 0]
y = data[:, 1]

#%%
plt.scatter(x, y, marker='x', c='r')
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profits in $10,000s')

#%% Cost function
def compute_cost(instance, label, theta):
    """

    :param instance: numpy array represents sample instances
    :param label: numpy array represents labels
    :param theta: feature weight
    :return: cost
    """
    m = np.size(label)
    print(m)
    # return 1 / (2 * m) * np.sum((instance.dot(theta).T - label) ** 2)  # somehow the label is a row vector
    return 1 / (2 * m) * np.sum((instance.dot(theta) - label.reshape(-1, 1)) ** 2)  # both work

#%%
x = data[:, 0]
x_biased = np.c_[np.ones(np.size(y)), x]
theta = np.array([[0], [0]])
compute_cost(x, y, theta)

#%%
def gradient_descent(sample, label, theta, alpha, iterations):
    print(label)
    m = len(label)
    for i in range(iterations):
        theta = theta - alpha * (1 / m) * sample.T.dot(sample.dot(theta) - label)
    return theta

#%%
theta_new = gradient_descent(x, y.reshape(-1,1), theta, 0.01, 1500)
plt.scatter(x[:, 1], y, marker='x', c='r')
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profits in $10,000s')
plt.plot(x[:, 1], x.dot(theta_new), 'b-')