"""Monte carlo sampling of regular arrays and then plot
(x1,y1,x2,y2,x3,y3) for 3 most important turbines"""
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append(r'/home/andrewkirby72/phd_work/data_synthesis')
from regular_array_sampling.functions import find_important_turbines


num_iters = 10000
turbine_coords = np.zeros(6)

fig = plt.figure(figsize=(10.0, 3.0))
turbine1 = fig.add_subplot(1, 3, 1)
turbine1.set_xlim([-5, 5])
turbine1.set_ylim([0, 30])
turbine1.set_xlabel('x_1 (D m)')
turbine1.set_ylabel('y_1 (D m)')
turbine2 = fig.add_subplot(1, 3, 2)
turbine2.set_xlim([-5, 5])
turbine2.set_ylim([0, 30])
turbine2.set_xlabel('x_2 (D m)')
turbine2.set_ylabel('y_2 (D m)')
turbine3 = fig.add_subplot(1, 3, 3)
turbine3.set_xlim([-5, 5])
turbine3.set_ylim([0, 30])
turbine3.set_xlabel('x_3 (D m)')
turbine3.set_ylabel('y_3 (D m)')

for i in range(num_iters):
    S_x = np.random.uniform(2, 20)
    S_y = np.random.uniform(2, 20)
    S_off = np.random.uniform(-S_x, S_x)
    theta = np.random.uniform(0, np.pi)
    turbine_coords = find_important_turbines(S_x, S_y, S_off, theta)
    # check if 3rd turbine is in x3 = [-5,5], y3 = [0,30]
    if np.all(np.abs(turbine_coords[:, 0]) < 5) \
            and np.all(turbine_coords[:, 1] < 30):
        turbine1.scatter(turbine_coords[0, 0], turbine_coords[0, 1])
        turbine2.scatter(turbine_coords[1, 0], turbine_coords[1, 1])
        turbine3.scatter(turbine_coords[2, 0], turbine_coords[2, 1])

plt.savefig('analysis/regular_array_monte_carlo_plots/regular_array_monte_carlo'\
    +str(num_iters)+'.png')
