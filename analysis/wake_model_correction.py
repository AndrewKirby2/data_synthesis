import matplotlib.pyplot as plt
import numpy as np


SxSy18 = np.array([[0.29, 0.4], [0.43, 0.44], [0.53, 0.46], [0.64, 0.46]])
SxSy27 = np.array([[0.31, 0.4], [0.44, 0.53], [0.59, 0.56], [0.54, 0.54], [0.67, 0.58], [0.74, 0.57]])
SxSy41 = np.array([[0.44, 0.53], [0.59, 0.62], [0.72, 0.65], [0.68, 0.64], [0.83, 0.64], [0.78, 0.67]])

plt.figure(0)
plt.scatter(SxSy18[:,0], SxSy18[:,1], label='S_x*S_y=18.3')
plt.scatter(SxSy27[:,0], SxSy27[:,1], label='S_x*S_y=27.4')
plt.scatter(SxSy41[:,0], SxSy41[:,1], label='S_x*S_y=41.1')
plt.xlabel('P_jensen/P_1')
plt.ylabel('P_inf/P_1')
plt.legend()
plt.savefig('jenson_correction.png')

v_def_square = np.array([0.0937, 0.0937, 0.0937, 0.022, 0.015, 0.0022, 0.001, 0.0, 0.0732])
SxSy = np.array([36, 18, 9, 36, 49, 18, 36, 54, 49])
CT_star = np.array([0.514, 0.680, 0.793, 0.824, 0.814, 0.842, 0.879, 0.899, 0.579])

plt.figure(1)
plt.scatter(SxSy, v_def_square, c=CT_star)
plt.xlabel('Sx*Sy (D^2 m^2)')
plt.ylabel('Sum of squared normalised velocity defects')
plt.colorbar()
plt.savefig('CT*against_veldef_and_density.png')