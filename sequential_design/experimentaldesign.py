"""Module containing class to make latinhypercube design
with valid 
"""
from scipy.stats import norm, expon
import numpy as np
import sys
from mogp_emulator.ExperimentalDesign import LatinHypercubeDesign
sys.path.append(r'/home/andrewkirby72/phd_work/data_synthesis')
from regular_array_sampling.functions import calculate_distance


def validlhs():
    from pyDOE import lhs as lhsd
    n_samples = 0
    while n_samples != 300:
        lhs = lhsd(6, 2800)
        #expand x, y coordinates to their real values
        lhs[:, 0] = expon(scale=10).ppf(lhs[:, 0])
        lhs[:, 1] = norm(0, 2.5).ppf(lhs[:, 1])
        lhs[:, 2] = expon(scale=10).ppf(lhs[:, 2])
        lhs[:, 3] = norm(0, 2.5).ppf(lhs[:, 3])
        lhs[:, 4] = expon(scale=10).ppf(lhs[:, 4])
        lhs[:, 5] = norm(0, 2.5).ppf(lhs[:, 5])
        #exclude points where turbine 2 is closer than turbine 1
        valid_1_2 = calculate_distance(lhs[:, 2],
                                        lhs[:, 3]) \
            - calculate_distance(lhs[:, 0], lhs[:, 1])
        lhs = lhs[valid_1_2 > 0]
        #exclude points where turbine 3 is closer than turbine 2
        valid_2_3 = calculate_distance(lhs[:, 4],
                                        lhs[:, 5]) \
            - calculate_distance(lhs[:, 2], lhs[:, 3])
        lhs = lhs[valid_2_3 > 0]
        #exclude points where turbines are closer than 2D to origin
        dist_1 = np.sqrt(lhs[:, 0]**2 + lhs[:, 1]**2)
        lhs = lhs[dist_1 > 2]
        dist_2 = np.sqrt(lhs[:, 2]**2 + lhs[:, 3]**2)
        lhs = lhs[dist_2 > 2]
        dist_3 = np.sqrt(lhs[:, 4]**2 + lhs[:, 5]**2)
        lhs = lhs[dist_3 > 2]
        #exclude points where turbines are closer than 2D from
        #each other
        dist_1_2 = np.sqrt((lhs[:, 0]-lhs[:, 2])**2 + 
                        (lhs[:, 1]-lhs[:, 3])**2)
        lhs = lhs[dist_1_2 > 2]
        dist_2_3 = np.sqrt((lhs[:, 2]-lhs[:, 4])**2 + 
                        (lhs[:, 3]-lhs[:, 5])**2)
        lhs = lhs[dist_2_3 > 2]
        dist_3_1 = np.sqrt((lhs[:, 4]-lhs[:, 0])**2 + 
                        (lhs[:, 5]-lhs[:, 1])**2)
        lhs = lhs[dist_3_1 > 2]
        n_samples = len(lhs)
        print(n_samples)
    #return to transformed coordinates
    lhs[:, 0] = expon(scale=10).cdf(lhs[:, 0])
    lhs[:, 1] = norm(0, 2.5).cdf(lhs[:, 1])
    lhs[:, 2] = expon(scale=10).cdf(lhs[:, 2])
    lhs[:, 3] = norm(0, 2.5).cdf(lhs[:, 3])
    lhs[:, 4] = expon(scale=10).cdf(lhs[:, 4])
    lhs[:, 5] = norm(0, 2.5).cdf(lhs[:, 5])
    return lhs

class validLCDesign(LatinHypercubeDesign):
    def __init__(self):
        LatinHypercubeDesign.__init__(self, 6)
    
    def _draw_samples(self, n_samples):
        if n_samples == 1:
            x1 = expon(scale=10).cdf(5)
            x2 = expon(scale=10).cdf(10)
            x3 = expon(scale=10).cdf(15)
            return np.array([[x1,0.5,x2,0.5,x3,0.5]])
        else:
            return validlhs()
