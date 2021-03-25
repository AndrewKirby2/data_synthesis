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
    farm_coords: ndarray of shape (49, 2)
        The x, y coordinates of the closest 49 wind turbines
    """
    farm_coords = np.zeros((49, 3))
    for n_y in np.arange(-3, 4):
        farm_coords[7*(n_y + 3):7*(n_y + 4), 0:2] = [calculate_turbine_coords(S_x, S_y,
            S_off, theta, n_x, n_y) for n_x in np.arange(-3, 4)]
    farm_coords[:,2] = calculate_distance(farm_coords[:,0],farm_coords[:,1])
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
