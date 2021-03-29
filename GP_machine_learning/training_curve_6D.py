import numpy as np
import matplotlib.pyplot as plt
from GP_machine_learning_functions import *
from regular_array_sampling.functions import regular_array_monte_carlo
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

rmse = np.ones((25,2))
noise = 0.01
cand_points = regular_array_monte_carlo(1000)

for n in range(25):
    n_target = 10 +n*160

    X_test, y_test = create_testing_points(noise)
    X_train, y_train, n_train = create_training_points_irregular(n_target, noise)

    #fit GP regression and calculate rmse
    kernel = 1.0 ** 2 * RBF(length_scale=[1.,1.,1.,1.,1.,1.]) + WhiteKernel(noise_level = 1e-5, noise_level_bounds=[1e-10,1])
    pipe = Pipeline([('scaler', StandardScaler()), ('gp', GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20))])
    pipe.fit(X_train, y_train-0.88)
    y_predict = pipe.predict(X_test)
    mse=mean_squared_error(y_test-0.88, y_predict)
    #report rmse
    print(n_train, np.sqrt(mse))
    rmse[n, 0] = n_train
    rmse[n, 1] = np.sqrt(mse)

plt.scatter(rmse[:,0], rmse[:,1])
plt.yscale('log')
plt.ylim([1e-3,1e-1])
plt.xlim([0,300])
plt.title('Training curve RBF - 6D 1% noise - Iregular array training')
plt.ylabel('RMSE')
plt.xlabel('Training points')
plt.savefig('gp_training_curve_RBF_irregular.png')