"""Module containing functions to simulate data from LES
"""
import numpy as np


def simulator2D(x, noise_level):
    """ calculates CT* based on x1, y1
        Gaussian noise is added

    Parameters
    ----------
    x : ndarray of shape (2,)
        The x, y coordinates of turbine 1

    Returns
    -------
    CT* : float
        The value of the turbine thrust coefficient
    """
    return 0.88 + 0.05*np.exp(-x[0]/10)*np.exp(-x[1]**2/(10+0.2*x[0])) \
        - 0.4*np.exp(-x[0]/10)*np.exp(-x[1]**2/(0.5+0.2*x[0])) \
        + np.random.normal(0, noise_level/1.96)


def simulator4d(x, noise_level):
    """ calculates CT* based on x1, y1, x2, y2
        Gaussian noise is added

    Parameters
    ----------
    x : ndarray of shape (4,)
        The x, y coordinates of turbines 1 and 2

    Returns
    -------
    CT* : float
        The value of the turbine thrust coefficient
    """
    return 0.88 + 0.05*np.exp(-x[0]/10)*np.exp(-x[1]**2/(10+0.2*x[0])) \
        - 0.4*np.exp(-x[0]/10)*np.exp(-x[1]**2/(0.5+0.2*x[0])) \
        + 0.05*np.exp(-x[2]/10)*np.exp(-x[3]**2/(10+0.2*x[2])) \
        - 0.4*np.exp(-x[2]/10)*np.exp(-x[3]**2/(0.5+0.2*x[2])) \
        + np.random.normal(0, noise_level/1.96)


def simulator6d(x, noise_level):
    """ calculates CT* based on x1, y1, x2, y2,
    x3, y3
        Gaussian noise is added

    Parameters
    ----------
    x : ndarray of shape (6,)
        The x, y coordinates of turbines 1, 2 and 3
        x is increasing in the upwind direction

    Returns
    -------
    CT* : float
        The value of the turbine thrust coefficient
    """
    return 0.88 + 0.05*np.exp(-x[0]/10)*np.exp(-x[1]**2/(10+0.2*x[0])) \
        - 0.4*np.exp(-x[0]/10)*np.exp(-x[1]**2/(0.5+0.2*x[0])) \
        + 0.05*np.exp(-x[2]/10)*np.exp(-x[3]**2/(10+0.2*x[2])) \
        - 0.4*np.exp(-x[2]/10)*np.exp(-x[3]**2/(0.5+0.2*x[2])) \
        + 0.05*np.exp(-x[4]/10)*np.exp(-x[5]**2/(10+0.2*x[4])) \
        - 0.4*np.exp(-x[4]/10)*np.exp(-x[5]**2/(0.5+0.2*x[4])) \
        + np.random.normal(0, noise_level/1.96)
