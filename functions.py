import numpy as np

def calculate_distance(x,y):
    """ calculate the effective distance of a wind turbine
    assuming the streamwise direction is 10x more important
    than spanwise direction
    distance = sqrt(10*x^2 + y^2)

    Parameters
    ----------
    x : float
        The spanwise coordinate of the turbine
    y: float
        The streamwise coordinate of the turbine
    
    Returns 
    -------
    distance : float
        The effective distance
    """
    distance = np.sqrt(10*x**2 + y**2)
    return distance