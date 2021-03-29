""" Module containing functions to calculate the 3 most
important wind turbines in a regular wind farm
"""
import numpy as np


def calculate_distance(x, y):
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
    distance[y < 0] = np.inf
    return distance


def make_wind_farm_coords(S_x, S_y, S_off, theta):
    """ calculate the x, y coordinates of a wind farm
    with regular arrangement

    Parameters
    ----------
    S_x: float
        The spanwise spacing of the wind farm
    S_y: float
        The streamwise spacing of the wind farm
    S_off : float
        The spanwise offset of row of turbines
    theta: float
        The angle of incoming wind

    Returns
    -------
    farm_coords: ndarray of shape (49, 3)
        (49, :3) The x, y coordinates of the closest 49 wind turbines
        (49, 3) The effective distance of the closest 49 wind turbines
    """
    farm_coords = np.zeros((49, 3))
    for n_y in np.arange(-3, 4):
        farm_coords[7*(n_y + 3):7*(n_y + 4), 0:2] = \
            [calculate_turbine_coords(S_x, S_y, S_off, theta, n_x, n_y)
             for n_x in np.arange(-3, 4)]
    farm_coords[:, 2] = calculate_distance(farm_coords[:, 0],
                                           farm_coords[:, 1])
    return farm_coords


def calculate_turbine_coords(S_x, S_y, S_off, theta, n_x, n_y):
    """caulcate the x, y coordinates of a single wind
    turbine in a regular arrangment

    Parameters
    ----------
    S_x: float
        The spanwise spacing of the wind farm
    S_y: float
        The streamwise spacing of the wind farm
    S_off : float
        The spanwise offset of row of turbines
    theta: float
        The angle of incoming wind
    n_x: integer
        Turbine number in original spanwise direction
    n_y: integer
        Turbine number in the original streamwise direction

    Returns
    -------
    turbine coords: tuple of shape (2,)
        The x, y coordinates of the wind turbine specified by
        the turbine numbers n_x, n_y
    """
    x = np.cos(theta)*S_x*n_x + np.cos(theta)*S_off*n_y \
        - np.sin(theta)*S_y*n_y
    y = np.sin(theta)*S_x*n_x + np.sin(theta)*S_off*n_y \
        + np.cos(theta)*S_y*n_y
    return (x, y)


def find_important_turbines(S_x, S_y, S_off, theta):
    """caulcate the x, y coordinates of the three closest
    turbines in a regular arrangment
    Distance is calculated assuming streamwise direction
    is 10x more important than spanwise direction

    Parameters
    ----------
    S_x: float
        The spanwise spacing of the wind farm
    S_y: float
        The streamwise spacing of the wind farm
    S_off : float
        The spanwise offset of row of turbines
    theta: float
        The angle of incoming wind

    Returns
    -------
    turbine coords: ndarray of shape(6,)
        The x, y coordinates of the three closest
    turbines in a regular arrangment
    """
    farm_coords = make_wind_farm_coords(S_x, S_y, S_off, theta)
    closest_turbines = np.argsort(farm_coords[:, 2])
    turbine_coords = farm_coords[closest_turbines[1:4], :2]
    turbine_coords = np.reshape(turbine_coords, 6)
    return turbine_coords

def regular_array_monte_carlo(n_samples):
    """ Monte carlo sample regular arrays in the range
    S_x = [2,40], S_y = [2,40], S_off = [0, S_x]
    theta = [0, pi]
    Only returns arrangements where the 3 turbines are in
    the domain x = [-5, 5] and y = [0,30]

    Parameters
    ----------
    n_samples: int
        The number of turbine arrangements sampled
    
    Returns
    -------
    turbine_coords: ndarray of shape (n_samples, 6)
    """
    layout_coords = np.zeros((n_samples, 6))
    sample_successes = 0
    while sample_successes < n_samples:
        S_x = np.random.uniform(2, 40)
        S_y = np.random.uniform(2, 40)
        S_off = np.random.uniform(0, S_x)
        theta = np.random.uniform(0, np.pi)
        turbine_coords = find_important_turbines(S_x, S_y, S_off, theta)
        # check if all turbines are in x3 = [-5,5], y3 = [0,30]
        if np.all(np.abs(turbine_coords[[0,2,4]]) < 5) \
            and np.all(turbine_coords[[1,3,5]] < 30):
            layout_coords[sample_successes, :] = turbine_coords
            sample_successes += 1
    return layout_coords