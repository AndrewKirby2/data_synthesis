import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)


#  First the noiseless case
X = np.array([3.])

# Observations
y = np.array([0.])

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.linspace(1, 5, 1000)

# Instantiate a Gaussian Process model
kernel = RBF(10, (1e-5, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X[:, np.newaxis], y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x[:, np.newaxis], return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure(1)
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill_between(x, y_pred - sigma, y_pred + sigma,
                     alpha=0.2, color='k')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend(loc='upper left')
plt.savefig('demo/skx.png')

#  First the noiseless case
X = np.array([2., 4., 5.])

# Observations
y = np.array([0., 0., 0.])

kernel = RBF(1, (1e-5, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

gp.fit(X[:, np.newaxis], y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x[:, np.newaxis], return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure(2)
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill_between(x, y_pred - sigma, y_pred + sigma,
                     alpha=0.2, color='k')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend(loc='upper left')
plt.savefig('demo/sGkUx.png')