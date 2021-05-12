import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm, expon
sys.path.append(r'/home/andrewkirby72/phd_work/data_synthesis')
from data_simulator.simulators import simulator2D

x = np.linspace(0,30,200)
y = np.linspace(-5,5,200)
x, y = np.meshgrid(x, y)
x = np.reshape(x, 40000)
y = np.reshape(y, 40000)
z = np.zeros(40000)

for i in range(len(x)):
    coords = np.array([x[i], y[i]])
    z[i] = simulator2D(coords, 0)

fig, ax = plt.subplots()
plt.scatter(expon(scale=10).cdf(x), norm(0, 2.5).cdf(y), c=z)
plt.xlabel('Transformed streamwise direction')
plt.ylabel('Transformed spanwise direction')
plt.colorbar()
plt.yticks(norm(0, 2.5).cdf(np.arange(-5,6,1)))
ax.set_yticklabels(np.arange(-5,6,1))
plt.xticks(expon(scale = 10).cdf(np.arange(0,31,5)))
ax.set_xticklabels(np.arange(0,31,5))
plt.savefig('analysis/GP_machine_learning_plots/transformed_simulator.png')
