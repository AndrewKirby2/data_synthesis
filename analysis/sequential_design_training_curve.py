import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon
from sklearn.metrics import mean_squared_error, mean_absolute_error
from palettable.cartocolors.diverging import Geyser_3
import mogp_emulator
import pickle
import sys
sys.path.append(r'/home/andrewkirby72/phd_work/data_synthesis')
from sequential_design.experimentaldesign import validLCDesign
from data_simulator.simulators import simulator6d_halved
from GP_machine_learning.GP_machine_learning_functions import *

validLHS = validLCDesign()
n_init = 1
n_samples = 100
n_cand = 300
md = mogp_emulator.MICEDesign(validLHS, simulator6d_halved, n_samples=n_samples, n_init=n_init, n_cand=n_cand)

init_design = md.generate_initial_design()
X_test, X_test_tran, y_test = create_testing_points_transformed()

x = np.zeros((101,6))
x[0, :] = init_design
x[0,0] = expon(scale=10).ppf(x[0, 0])
x[0,2] = expon(scale=10).ppf(x[0, 2])
x[0,4] = expon(scale=10).ppf(x[0, 4])
x[0,1] = norm(0, 2.5).ppf(x[0, 1])
x[0,3] = norm(0, 2.5).ppf(x[0, 3])
x[0,5] = norm(0, 2.5).ppf(x[0, 5])

init_target = simulator6d_halved(x[0, :])
md.set_initial_targets(init_target)
mae = np.zeros(100)
rmse = np.zeros(100)

for d in range(n_samples):
    next_point = md.get_next_point()
    x[d+1] = next_point
    x[d+1,0] = expon(scale=10).ppf(x[d+1, 0])
    x[d+1,2] = expon(scale=10).ppf(x[d+1, 2])
    x[d+1,4] = expon(scale=10).ppf(x[d+1, 4])
    x[d+1,1] = norm(0, 2.5).ppf(x[d+1, 1])
    x[d+1,3] = norm(0, 2.5).ppf(x[d+1, 3])
    x[d+1,5] = norm(0, 2.5).ppf(x[d+1, 5])
    next_target = simulator6d_halved(x[d+1,:])
    print(x[d+1, :])
    print(next_target)
    md.set_next_target(next_target)

    X_train = x
    inputs = md.get_inputs()
    targets = md.get_targets()

    gp_mice = mogp_emulator.GaussianProcess(inputs, targets)
    gp_mice = mogp_emulator.fit_GP_MAP(inputs, targets)

    y_predict = gp_mice(X_test_tran)
    rmse[d] = np.sqrt(mean_squared_error(y_test, y_predict))
    mae[d] = mean_absolute_error(y_test, y_predict)


max_error = np.max(np.abs((y_predict - y_test)))

plt.figure(1)
plt.scatter(np.arange(2,102,1), mae)
plt.ylabel('MAE')
plt.xlabel('Number of training points')
plt.savefig('analysis/sequential_design_plots/seq_design_mae_regular_LHS.png')

plt.figure(2)
plt.scatter(np.arange(2,102,1), rmse)
plt.ylabel('RMSE')
plt.xlabel('Number of training points')
plt.savefig('analysis/sequential_design_plots/seq_design_rmse_regular_LHS.png')

np.savetxt('inputs.txt', inputs)
