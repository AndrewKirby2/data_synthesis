""" Plot the testing errors for the 6D data simulator of CT*
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from palettable.cartocolors.diverging import Geyser_3
import sys
sys.path.append(r'/home/andrewkirby72/phd_work/data_synthesis')
from GP_machine_learning.GP_machine_learning_functions import *
from regular_array_sampling.functions import regular_array_monte_carlo

noise = 0.01
# create array of sampled regular array layouts
#cand_points = regular_array_monte_carlo(10000)
# create testing points
X_test, y_test = create_testing_points(noise)

n_target = 500

# create training points
X_train, y_train, n_train = \
    create_training_points_irregular_lhs(n_target, noise)

# fit GP regression and calculate rmse
kernel = 1.0 ** 2 * Matern(length_scale=[1., 1., 1., 1., 1., 1.]) \
    + WhiteKernel(noise_level=1e-5, noise_level_bounds=[1e-10, 1])
pipe = Pipeline([('scaler', StandardScaler()),
                ('gp', GaussianProcessRegressor(kernel=kernel,
                    n_restarts_optimizer=20))])
pipe.fit(X_train, y_train)
y_predict = pipe.predict(X_test)
mse = mean_squared_error(y_test, y_predict)
# report rmse
print(n_train, np.sqrt(mse))

default_mse = mean_squared_error(y_test, np.zeros(len(y_test)))
print(np.sqrt(default_mse))

max_error = np.max(np.abs((y_predict - y_test)))

fig = plt.figure(figsize=(12.0, 5.0))
turbine1 = fig.add_subplot(1, 3, 1)
turbine1.set_xlabel('x_1 (D m)')
turbine1.set_ylabel('y_1 (D m)')
turbine2 = fig.add_subplot(1, 3, 2)
turbine2.set_xlabel('x_2 (D m)')
turbine2.set_ylabel('y_2 (D m)')
turbine3 = fig.add_subplot(1, 3, 3)
turbine3.set_xlabel('x_3 (D m)')
turbine3.set_ylabel('y_3 (D m)')

x = turbine1.scatter(X_test[:, 0], X_test[:, 1],
                 c=(y_predict - y_test),cmap=Geyser_3.mpl_colormap, vmin=-max_error, vmax=max_error)
turbine1.scatter(X_train[:, 0], X_train[:, 1], c='black')
turbine2.scatter(X_test[:, 2], X_test[:, 3],
                 c=(y_predict - y_test),cmap=Geyser_3.mpl_colormap, vmin=-max_error, vmax=max_error)
turbine2.scatter(X_train[:, 2], X_train[:, 3], c='black')
turbine3.scatter(X_test[:, 4], X_test[:, 5],
                 c=(y_predict - y_test),cmap=Geyser_3.mpl_colormap, vmin=-max_error, vmax=max_error)
turbine3.scatter(X_train[:, 4], X_train[:, 5], c='black')
plt.colorbar(x)
plt.savefig('analysis/GP_machine_learning_plots/GP_error_irregular_training_max_change_halved.png')
