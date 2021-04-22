"""Module containing helper functions for Gaussian
Process machine learning
"""
from pyDOE import lhs
from scipy.stats.distributions import norm, expon, uniform
import diversipy.hycusampling as dp
import diversipy.subset as sb
from scipy.stats import norm, expon
import numpy as np
import sys
sys.path.append(r'/home/andrewkirby72/phd_work/data_synthesis')
from data_simulator.simulators import simulator6d_halved
from regular_array_sampling.functions import regular_array_monte_carlo
from regular_array_sampling.functions import find_important_turbines
from regular_array_sampling.functions import calculate_distance


def create_testing_points(noise_level):
    """ create array of testing points
    Discard any training points where turbines are
    not in the correct order and any training points where
    turbines are closer than 2D

    Parameters
    ----------
    noise_level: float
        Level of gaussian noise to be added to
        simulator

    Returns
    -------
    X_test_real:    ndarray of shape(variable,6)
                    array containing valid test points
    y_test:         ndarray of shape(variable,)
                    value of CT* at test points
    """
    X_test = lhs(6, 1000, 'maximin')
    X_test[:, 0] = 30*X_test[:, 0]
    X_test[:, 1] = 10*X_test[:, 1] - 5
    X_test[:, 2] = 30*X_test[:, 2]
    X_test[:, 3] = 10*X_test[:, 3] - 5
    X_test[:, 4] = 30*X_test[:, 4]
    X_test[:, 5] = 10*X_test[:, 5] - 5
    # exclude test points where turbine 1 is closer than 2D
    X_test_dist = np.sqrt(X_test[:, 0]**2 + X_test[:, 1]**2)
    X_test_real = X_test[X_test_dist > 2]
    # exclude test points where turbine 2 is more "important" than turbine 1
    # using distance = sqrt(x_1^2 + k*y_1^2)
    X_test_sig = calculate_distance(X_test_real[:, 2],
                                    X_test_real[:, 3]) \
        - calculate_distance(X_test_real[:, 0], X_test_real[:, 1])
    X_test_real = X_test_real[X_test_sig > 0]
    # exclude test points where turbine 3 is more "important" than turbine 2
    # using distance = sqrt(x_1^2 + k*y_1^2)
    X_test_sig = calculate_distance(X_test_real[:, 4],
                                    X_test_real[:, 5]) \
        - calculate_distance(X_test_real[:, 2], X_test_real[:, 3])
    X_test_real = X_test_real[X_test_sig > 0]
    y_test = np.zeros(len(X_test_real))
    for i in range(len(X_test_real)):
        y_test[i] = simulator6d_halved(X_test_real[i, :], noise_level)
    return X_test_real, y_test


def create_training_points_irregular(n_target, noise_level):
    """ create array of training points
    Discard any training points where turbines are
    not in the correct order and any training points where
    turbines are closer than 2D

    Parameters
    ----------
    n_target: int
        target number of training points
    noise_level: float
        Level of gaussian noise to be added to
        simulator

    Returns
    -------
    X_train_real:    ndarray of shape(variable,6)
                    array containing valid training points
    y_train:         ndarray of shape(variable,)
                    value of CT* at test points
    n_train:        int
                    number of valid training points
    """
    X_train = dp.maximin_reconstruction(n_target, 6)
    X_train[:, 0] = 30*X_train[:, 0]
    X_train[:, 1] = 10*X_train[:, 1] - 5
    X_train[:, 2] = 30*X_train[:, 2]
    X_train[:, 3] = 10*X_train[:, 3] - 5
    X_train[:, 4] = 30*X_train[:, 4]
    X_train[:, 5] = 10*X_train[:, 5] - 5
    # exclude training points where turbine 1 is closer than 2D
    X_train_dist = np.sqrt(X_train[:, 0]**2 + X_train[:, 1]**2)
    X_train_real = X_train[X_train_dist > 2]
    # exclude training points where turbine 2 is more important"
    # than turbine 1 using distance = sqrt(10*x_1^2 + y_1^2)
    X_train_sig = calculate_distance(X_train_real[:, 2],
                                    X_train_real[:, 3]) \
        - calculate_distance(X_train_real[:, 0], X_train_real[:, 1])
    X_train_real = X_train_real[X_train_sig > 0]
    # exclude training points where turbine 3 is more important
    # than turbine 2 using distance = sqrt(10*x_1^2 + y_1^2)
    X_train_sig = calculate_distance(X_train_real[:, 4],
                                    X_train_real[:, 5]) \
        - calculate_distance(X_train_real[:, 2], X_train_real[:, 3])
    X_train_real = X_train_real[X_train_sig > 0]
    # run simulations to find data points
    y_train = np.zeros(len(X_train_real))
    for i in range(len(X_train_real)):
        y_train[i] = simulator6d_halved(X_train_real[i, :], noise_level)
    n_train = len(X_train_real)
    return X_train_real, y_train, n_train

def create_training_points_irregular_lhs(n_target, noise_level):
    """ create array of training points
    Discard any training points where turbines are
    not in the correct order and any training points where
    turbines are closer than 2D
    Scale the training points using a gaussian for spanwise
    direction and exponential function for streamwise
    direction

    Parameters
    ----------
    n_target: int
        target number of training points
    noise_level: float
        Level of gaussian noise to be added to
        simulator

    Returns
    -------
    X_train_real:    ndarray of shape(variable,6)
                    array containing valid training points
    y_train:         ndarray of shape(variable,)
                    value of CT* at test points
    n_train:        int
                    number of valid training points
    """
    X_train = lhs(6, n_target, 'maximin')
    X_train[:, 0] = expon(scale = 7).ppf(X_train[:, 0])
    X_train[:, 1] = norm(scale = 1.5).ppf(X_train[:, 1])
    X_train[:, 2] = expon(scale = 7).ppf(X_train[:, 2])
    X_train[:, 3] = norm(scale = 1.5).ppf(X_train[:, 3])
    X_train[:, 4] = expon(scale = 7).ppf(X_train[:, 4])
    X_train[:, 5] = norm(scale = 1.5).ppf(X_train[:, 5])
    # exclude training points where turbine 1 is closer than 2D
    X_train_dist = np.sqrt(X_train[:, 0]**2 + X_train[:, 1]**2)
    X_train_real = X_train[X_train_dist > 2]
    # exclude training points where turbine 2 is more important"
    # than turbine 1 using distance = sqrt(10*x_1^2 + y_1^2)
    X_train_sig = calculate_distance(X_train_real[:, 2],
                                    X_train_real[:, 3]) \
        - calculate_distance(X_train_real[:, 0], X_train_real[:, 1])
    X_train_real = X_train_real[X_train_sig > 0]
    # exclude training points where turbine 3 is more important
    # than turbine 2 using distance = sqrt(10*x_1^2 + y_1^2)
    X_train_sig = calculate_distance(X_train_real[:, 4],
                                    X_train_real[:, 5]) \
        - calculate_distance(X_train_real[:, 2], X_train_real[:, 3])
    X_train_real = X_train_real[X_train_sig > 0]
    # run simulations to find data points
    y_train = np.zeros(len(X_train_real))
    for i in range(len(X_train_real)):
        y_train[i] = simulator6d_halved(X_train_real[i, :], noise_level)
    n_train = len(X_train_real)
    return X_train_real, y_train, n_train


def create_training_points_regular(n_target, noise_level, cand_points):
    """ create array of training points from
    regular turbine arrays

    Returns
    -------
    X_train_real:    ndarray of shape(variable,6)
                    array containing valid training points
    y_train:         ndarray of shape(variable,)
                    value of CT* at test points
    n_train:        int
                    number of valid training points
    """
    X_train_real = sb.select_greedy_maximin(cand_points, n_target)
    y_train = np.zeros(len(X_train_real))
    for i in range(len(X_train_real)):
        y_train[i] = simulator6d_halved(X_train_real[i, :], noise_level)
    n_train = n_target
    return X_train_real, y_train, n_train


def create_testing_points_regular(noise_level):
    """ create array of testing points from regular
    wind turbine arrays
    Discard any training points where turbines are
    not in the correct order and any training points where
    turbines are closer than 2D

    Parameters
    ----------
    noise_level: float
        Level of gaussian noise to be added to
        simulator

    Returns
    -------
    X_test_real:    ndarray of shape(variable,6)
                    array containing valid test points
    y_test:         ndarray of shape(variable,)
                    value of CT* at test points
    """
    X_test_real = regular_array_monte_carlo(1000)
    y_test = np.zeros(len(X_test_real))
    for i in range(len(X_test_real)):
        y_test[i] = simulator6d_halved(X_test_real[i, :], noise_level)
    return X_test_real, y_test

def create_training_points_regular_maxi4d(n_target, noise_level):
    """ create array of training points from
    regular turbine arrays
    Use maximin in 4d
    Returns
    -------
    X_train_real:    ndarray of shape(variable,6)
                    array containing valid training points
    y_train:         ndarray of shape(variable,)
                    value of CT* at test points
    n_train:        int
                    number of valid training points
    """
    regular_array = dp.maximin_reconstruction(n_target, 4)
    # rescale to design in range S_x = [2,20] S_y = [2,20],
    # S_off = [0, S_y] and theta = [0, pi]
    regular_array[:, 0] = 2 + 18*regular_array[:, 0]
    regular_array[:, 1] = 2 + 18*regular_array[:, 1]
    regular_array[:, 2] = regular_array[:, 1]*regular_array[:, 2]
    regular_array[:, 3] = np.pi*regular_array[:, 3]
    #convert regular array into 3 most important turbines
    X_train_real = np.zeros((n_target, 6))
    for i in range(n_target):
        X_train_real[i,:] = find_important_turbines(regular_array[i, 0],
                                                      regular_array[i, 1],
                                                      regular_array[i, 2],
                                                      regular_array[i, 3])
    y_train = np.zeros(len(X_train_real))
    for i in range(len(X_train_real)):
        y_train[i] = simulator6d_halved(X_train_real[i, :], noise_level)
    n_train = n_target
    return X_train_real, y_train, n_train

def create_testing_points_transformed():
    """ create array of testing points
    Discard any training points where turbines are
    not in the correct order and any training points where
    turbines are closer than 2D
    X_test is tranformed by the cdf of probability distributions
    expon(scale=10) in the x direction and norm(0, 2.5) in the y
    direction

    Parameters
    ----------
    noise_level: float
        Level of gaussian noise to be added to
        simulator

    Returns
    -------
    X_test:         ndarray of shape(variable,6)
                    array containing valid test points
    X_test_tran:    ndarray of shape(varaible,6)
                    array containing valid transformed test points
    y_test:         ndarray of shape(variable,)
                    value of CT* at test points
    """
    X_test = lhs(6, 10000)
    X_test[:, 0] = 30*X_test[:, 0]
    X_test[:, 1] = 10*X_test[:, 1] - 5
    X_test[:, 2] = 30*X_test[:, 2]
    X_test[:, 3] = 10*X_test[:, 3] - 5
    X_test[:, 4] = 30*X_test[:, 4]
    X_test[:, 5] = 10*X_test[:, 5] - 5
    # exclude test points where turbine 1 is closer than 2D
    X_test_dist = np.sqrt(X_test[:, 0]**2 + X_test[:, 1]**2)
    X_test_real = X_test[X_test_dist > 2]
    # exclude test points where turbine 2 is more "important" than turbine 1
    # using distance = sqrt(x_1^2 + k*y_1^2)
    X_test_sig = calculate_distance(X_test_real[:, 2],
                                    X_test_real[:, 3]) \
        - calculate_distance(X_test_real[:, 0], X_test_real[:, 1])
    X_test_real = X_test_real[X_test_sig > 0]
    # exclude test points where turbine 3 is more "important" than turbine 2
    # using distance = sqrt(x_1^2 + k*y_1^2)
    X_test_sig = calculate_distance(X_test_real[:, 4],
                                    X_test_real[:, 5]) \
        - calculate_distance(X_test_real[:, 2], X_test_real[:, 3])
    X_test_real = X_test_real[X_test_sig > 0]
    y_test = np.zeros(len(X_test_real))
    for i in range(len(X_test_real)):
        y_test[i] = simulator6d_halved(X_test_real[i, :])
    X_test = X_test_real
    X_test_tran = np.zeros((len(X_test_real), 6))
    X_test_tran[:, 0] = expon(scale=10).cdf(X_test_real[:, 0])
    X_test_tran[:, 2] = expon(scale=10).cdf(X_test_real[:, 2])
    X_test_tran[:, 4] = expon(scale=10).cdf(X_test_real[:, 4])
    X_test_tran[:, 1] = norm(0, 2.5).cdf(X_test_real[:, 1])
    X_test_tran[:, 3] = norm(0, 2.5).cdf(X_test_real[:, 3])
    X_test_tran[:, 5] = norm(0, 2.5).cdf(X_test_real[:, 5])
    return X_test, X_test_tran, y_test

def create_training_points_irregular_transformed(n_target, noise_level):
    """ create array of training points
    Discard any training points where turbines are
    not in the correct order and any training points where
    turbines are closer than 2D
    Maximin design in the transformed space

    Parameters
    ----------
    n_target: int
        target number of training points
    noise_level: float
        Level of gaussian noise to be added to
        simulator

    Returns
    -------
    X_train:        ndarray of shape(variable,6)
                    array containing valid training points
    X_train_tran:   ndarray of shape(variable,6)
                    array containing valid transformed training points
    y_train:         ndarray of shape(variable,)
                    value of CT* at test points
    n_train:        int
                    number of valid training points
    """
    X_train_tran = dp.maximin_reconstruction(n_target, 6)
    X_train = np.zeros((len(X_train_tran), 6))
    X_train[:, 0] = expon(scale=10).ppf(X_train_tran[:, 0])
    X_train[:, 1] = norm(0, 2.5).ppf(X_train_tran[:, 1])
    X_train[:, 2] = expon(scale=10).ppf(X_train_tran[:, 2])
    X_train[:, 3] = norm(0, 2.5).ppf(X_train_tran[:, 3])
    X_train[:, 4] = expon(scale=10).ppf(X_train_tran[:, 4])
    X_train[:, 5] = norm(0, 2.5).ppf(X_train_tran[:, 5])
    # exclude training points where turbine 1 is closer than 2D
    X_train_dist = np.sqrt(X_train[:, 0]**2 + X_train[:, 1]**2)
    X_train_real = X_train[X_train_dist > 2]
    X_train_tran = X_train_tran[X_train_dist > 2]
    # exclude training points where turbine 2 is more important"
    # than turbine 1 using distance = sqrt(10*x_1^2 + y_1^2)
    X_train_sig = calculate_distance(X_train_real[:, 2],
                                    X_train_real[:, 3]) \
        - calculate_distance(X_train_real[:, 0], X_train_real[:, 1])
    X_train_real = X_train_real[X_train_sig > 0]
    X_train_tran = X_train_tran[X_train_sig > 0]
    # exclude training points where turbine 3 is more important
    # than turbine 2 using distance = sqrt(10*x_1^2 + y_1^2)
    X_train_sig = calculate_distance(X_train_real[:, 4],
                                    X_train_real[:, 5]) \
        - calculate_distance(X_train_real[:, 2], X_train_real[:, 3])
    X_train_real = X_train_real[X_train_sig > 0]
    X_train_tran = X_train_tran[X_train_sig > 0]
    # run simulations to find data points
    y_train = np.zeros(len(X_train_real))
    for i in range(len(X_train_real)):
        y_train[i] = simulator6d_halved(X_train_real[i, :], noise_level)
    n_train = len(X_train_real)
    X_train = X_train_real
    return X_train, X_train_tran, y_train, n_train

def create_testing_points_regular_transformed():
    """ create array of testing points from regular
    wind turbine arrays
    Discard any training points where turbines are
    not in the correct order and any training points where
    turbines are closer than 2D

    Parameters
    ----------
    noise_level: float
        Level of gaussian noise to be added to
        simulator

    Returns
    -------
    X_test:         ndarray of shape(variable,6)
                    array containing valid test points
    X_test_tran:    ndarray of shape(variable,6)
                    array containing valid transformed test points
    y_test:         ndarray of shape(variable,)
                    value of CT* at test points
    """
    X_test_real = regular_array_monte_carlo(1000)
    y_test = np.zeros(len(X_test_real))
    for i in range(len(X_test_real)):
        y_test[i] = simulator6d_halved(X_test_real[i, :])
    X_test = X_test_real
    X_test_tran = np.zeros((1000, 6))
    X_test_tran[:, 0] = expon(scale=10).cdf(X_test_real[:, 0])
    X_test_tran[:, 2] = expon(scale=10).cdf(X_test_real[:, 2])
    X_test_tran[:, 4] = expon(scale=10).cdf(X_test_real[:, 4])
    X_test_tran[:, 1] = norm(0, 2.5).cdf(X_test_real[:, 1])
    X_test_tran[:, 3] = norm(0, 2.5).cdf(X_test_real[:, 3])
    X_test_tran[:, 5] = norm(0, 2.5).cdf(X_test_real[:, 5])
    return X_test, X_test_tran, y_test

def create_training_points_regular_transformed(n_target, noise_level, cand_points):
    """ create array of training points from
    regular turbine arrays

    Returns
    -------
    X_train:        ndarray of shape(variable,6)
                    array containing valid training points
    X_train_tran:   ndarray of shape(variable,6)
                    array containing valid transformed training points
    y_train:         ndarray of shape(variable,)
                    value of CT* at test points
    n_train:        int
                    number of valid training points
    """
    cand_points_tran = np.zeros((len(cand_points), 6))
    cand_points_tran[:, 0] = expon(scale=10).cdf(cand_points[:, 0])
    cand_points_tran[:, 2] = expon(scale=10).cdf(cand_points[:, 2])
    cand_points_tran[:, 4] = expon(scale=10).cdf(cand_points[:, 4])
    cand_points_tran[:, 1] = norm(0, 2.5).cdf(cand_points[:, 1])
    cand_points_tran[:, 3] = norm(0, 2.5).cdf(cand_points[:, 3])
    cand_points_tran[:, 5] = norm(0, 2.5).cdf(cand_points[:, 5])
    X_train_tran = sb.select_greedy_maximin(cand_points_tran, n_target)
    X_train = np.zeros((len(X_train_tran), 6))
    X_train[:, 0] = expon(scale=10).ppf(X_train_tran[:, 0])
    X_train[:, 1] = norm(0, 2.5).ppf(X_train_tran[:, 1])
    X_train[:, 2] = expon(scale=10).ppf(X_train_tran[:, 2])
    X_train[:, 3] = norm(0, 2.5).ppf(X_train_tran[:, 3])
    X_train[:, 4] = expon(scale=10).ppf(X_train_tran[:, 4])
    X_train[:, 5] = norm(0, 2.5).ppf(X_train_tran[:, 5])
    y_train = np.zeros(len(X_train))
    for i in range(len(X_train)):
        y_train[i] = simulator6d_halved(X_train[i, :], noise_level)
    n_train = n_target
    return X_train, X_train_tran, y_train, n_train