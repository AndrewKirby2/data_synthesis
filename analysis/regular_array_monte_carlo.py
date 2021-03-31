"""Monte carlo sampling of regular arrays and then plot
(x1,y1,x2,y2,x3,y3) for 3 most important turbines"""
import sys
import matplotlib.pyplot as plt
sys.path.append(r'/home/andrewkirby72/phd_work/data_synthesis')
from regular_array_sampling.functions import regular_array_monte_carlo


num_iters = 10000

fig = plt.figure(figsize=(12.0, 5.0))
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

turbine_coords = regular_array_monte_carlo(num_iters)
turbine1.scatter(turbine_coords[:, 0], turbine_coords[:, 1])
turbine2.scatter(turbine_coords[:, 2], turbine_coords[:, 3])
turbine3.scatter(turbine_coords[:, 4], turbine_coords[:, 5])

plt.savefig('analysis/regular_array_monte_carlo_plots/regular_array_monte_carlo10000.png')
