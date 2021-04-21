import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon
from sklearn.metrics import mean_squared_error, mean_absolute_error
from palettable.cartocolors.diverging import Geyser_3
import mogp_emulator
import sys
sys.path.append(r'/home/andrewkirby72/phd_work/data_synthesis')
from sequential_design.experimentaldesign import validLCDesign
from data_simulator.simulators import simulator6d_halved
from GP_machine_learning.GP_machine_learning_functions import *

validLHS = validLCDesign()
n_init = 1
n_samples = 60
n_cand = 300
md = mogp_emulator.MICEDesign(validLHS, simulator6d_halved, n_samples=n_samples, n_init=n_init, n_cand=n_cand)

init_design = md.generate_initial_design()

x = np.zeros((61,6))
x[0, :] = init_design
x[0,0] = expon(scale=10).ppf(x[0, 0])
x[0,2] = expon(scale=10).ppf(x[0, 2])
x[0,4] = expon(scale=10).ppf(x[0, 4])
x[0,1] = norm(0, 2.5).ppf(x[0, 1])
x[0,3] = norm(0, 2.5).ppf(x[0, 3])
x[0,5] = norm(0, 2.5).ppf(x[0, 5])

init_target = simulator6d_halved(x[0, :])
md.set_initial_targets(init_target)

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

X_test, X_test_tran, y_test = create_testing_points_transformed()

y_predict = gp_mice(X_test_tran)
mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
print('RMSE: ', np.sqrt(mse), ' MAE: ', mae)

max_error = np.max(np.abs((y_predict - y_test)))

fig = plt.figure(figsize=(12.0, 5.0))
turbine1 = fig.add_subplot(1, 3, 1)
turbine1.set_xlabel('x_1 (D m)')
turbine1.set_ylabel('y_1 (D m)')
turbine1.set_xlim([0, 30])
turbine1.set_ylim([-5 ,5])
turbine2 = fig.add_subplot(1, 3, 2)
turbine2.set_xlabel('x_2 (D m)')
turbine2.set_ylabel('y_2 (D m)')
turbine2.set_xlim([0, 30])
turbine2.set_ylim([-5 ,5])
turbine3 = fig.add_subplot(1, 3, 3)
turbine3.set_xlabel('x_3 (D m)')
turbine3.set_ylabel('y_3 (D m)')
turbine3.set_xlim([0, 30])
turbine3.set_ylim([-5 ,5])

x = turbine1.scatter(X_test[:, 0], X_test[:, 1],
                 c=(y_predict - y_test)/(y_test+0.88),cmap=Geyser_3.mpl_colormap, vmin=-0.1, vmax=0.1)
turbine1.scatter(X_train[:, 0], X_train[:, 1], c='black')
turbine2.scatter(X_test[:, 2], X_test[:, 3],
                 c=(y_predict - y_test)/(y_test+0.88),cmap=Geyser_3.mpl_colormap, vmin=-0.1, vmax=0.1)
turbine2.scatter(X_train[:, 2], X_train[:, 3], c='black')
turbine3.scatter(X_test[:, 4], X_test[:, 5],
                 c=(y_predict - y_test)/(y_test+0.88),cmap=Geyser_3.mpl_colormap, vmin=-0.1, vmax=0.1)
turbine3.scatter(X_train[:, 4], X_train[:, 5], c='black')
plt.colorbar(x)
plt.savefig('analysis/sequential_design_plots/seq_design_transformed_max_change_halved.png')

