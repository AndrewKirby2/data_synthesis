import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'/home/andrewkirby72/phd_work/data_synthesis')
from GP_machine_learning.GP_machine_learning_functions import *

inputs = np.loadtxt('inputs.txt')
X, X_tran, y = create_testing_points_regular_transformed()

min_dist = np.zeros(len(inputs))

for i in range(len(inputs)):
    diff = np.linalg.norm(X_tran - inputs[i, :], axis = 1)
    min_dist[i] = np.min(diff)

print(np.mean(min_dist))

plt.figure()
plt.scatter(np.arange(len(inputs)), min_dist)
plt.xlabel('Training point number')
plt.ylabel('Minimum distance from regular turbine array')
plt.savefig('analysis/sequential_design_plots/training_points_dist_from_regular.png')