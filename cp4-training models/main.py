#%% Imports
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

#%% Generate Random dataset
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)
plt.scatter(x, y)

# plt.show()

#%% Normal Equation for best fitting theta
x_b = np.c_[np.ones((100, 1)), x]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)

#%% Making prediction
x_new = np.array([[0], [2]])
x_new_b = np.c_[np.ones((2, 1)), x_new]
y_predict = x_new_b.dot(theta_best)


#%% Plot predictions
plt.plot(x_new, y_predict, "r-")
plt.plot(x, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

#%% SKlearn linear regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)
print("x0:{}, coef: {}".format(lin_reg.intercept_, lin_reg.coef_))
print("Prediction", lin_reg.predict(x_new))

