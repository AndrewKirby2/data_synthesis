import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'/home/andrewkirby72/phd_work/data_synthesis')
from sequential_design.experimentaldesign import validlhs

lhs = validlhs()
print(len(lhs))

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

turbine1.scatter(lhs[:, 0], lhs[:, 1], c='black')
turbine2.scatter(lhs[:, 2], lhs[:, 3], c='black')
turbine3.scatter(lhs[:, 4], lhs[:, 5], c='black')

plt.savefig('analysis/sequential_design_plots/cand_points.png')
