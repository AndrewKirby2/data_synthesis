"""Maximin sampling of regular arrays and then plot
(x1,y1,x2,y2,x3,y3) for 3 most important turbines"""
import sys
import matplotlib.pyplot as plt
import diversipy.hycusampling as dp
import numpy as np
sys.path.append(r'/home/andrewkirby72/phd_work/data_synthesis')
from regular_array_sampling.functions import find_important_turbines

num_points = 60
regular_array = dp.maximin_reconstruction(num_points, 4)
# rescale to design in range S_x = [2,20] S_y = [2,20],
# S_off = [0, S_x] and theta = [0, pi]
regular_array[:, 0] = 2 + 18*regular_array[:, 0]
regular_array[:, 1] = 2 + 18*regular_array[:, 1]
regular_array[:, 2] = regular_array[:, 0]*regular_array[:, 2]
regular_array[:, 3] = np.pi*regular_array[:, 3]

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

turbine_coords = np.zeros((num_points, 6))
for i in range(num_points):
    turbine_coords[i,:] = find_important_turbines(regular_array[i, 0],
                                         regular_array[i, 1],
                                         regular_array[i, 2],
                                         regular_array[i, 3])

turbine1.scatter(turbine_coords[:, 0], turbine_coords[:, 1])
turbine2.scatter(turbine_coords[:, 2], turbine_coords[:, 3])
turbine3.scatter(turbine_coords[:, 4], turbine_coords[:, 5])

plt.savefig('analysis/regular_array_monte_carlo_plots/regular_array_40.png')
