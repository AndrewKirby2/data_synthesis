"""Module containing helper functions for Gaussian
Process machine learning
"""
from pyDOE import lhs
import diversipy.hycusampling as dp
import diversipy.subset as sb
import numpy as np
import sys
sys.path.append(r'/home/andrewkirby72/phd_work/data_synthesis')
from data_simulator.simulators import simulator6d
from regular_array_sampling.functions import regular_array_monte_carlo


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
    X_test[:, 0] = 10*X_test[:, 0] - 5
    X_test[:, 1] = 30*X_test[:, 1]
    X_test[:, 2] = 10*X_test[:, 2] - 5
    X_test[:, 3] = 30*X_test[:, 3]
    X_test[:, 4] = 10*X_test[:, 4] - 5
    X_test[:, 5] = 30*X_test[:, 5]
    # exclude test points where turbine 1 is closer than 2D
    X_test_dist = np.sqrt(X_test[:, 0]**2 + X_test[:, 1]**2)
    X_test_real = X_test[X_test_dist > 2]
    # exclude test points where turbine 2 is more "important" than turbine 1
    # using distance = sqrt(10*x_1^2 + y_1^2)
    X_test_sig = np.sqrt(np.sqrt(10*X_test_real[:, 2]**2
                         + X_test_real[:, 3]**2)) \
        - np.sqrt(np.sqrt(10*X_test_real[:, 0]**2 + X_test_real[:, 1]**2))
    X_test_real = X_test_real[X_test_sig > 0]
    # exclude test points where turbine 3 is more "important" than turbine 2
    # using distance = sqrt(10*x_1^2 + y_1^2)
    X_test_sig = np.sqrt(np.sqrt(10*X_test_real[:, 4]**2
                         + X_test_real[:, 5]**2)) \
        - np.sqrt(np.sqrt(10*X_test_real[:, 2]**2 + X_test_real[:, 3]**2))
    X_test_real = X_test_real[X_test_sig > 0]
    y_test = np.zeros(len(X_test_real))
    for i in range(len(X_test_real)):
        y_test[i] = simulator6d(X_test_real[i, :], noise_level)
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
    X_train[:, 0] = 10*X_train[:, 0] - 5
    X_train[:, 1] = 30*X_train[:, 1]
    X_train[:, 2] = 10*X_train[:, 2] - 5
    X_train[:, 3] = 30*X_train[:, 3]
    X_train[:, 4] = 10*X_train[:, 4] - 5
    X_train[:, 5] = 30*X_train[:, 5]
    # exclude training points where turbine 1 is closer than 2D
    X_train_dist = np.sqrt(X_train[:, 0]**2 + X_train[:, 1]**2)
    X_train_real = X_train[X_train_dist > 2]
    # exclude training points where turbine 2 is more important"
    # than turbine 1 using distance = sqrt(10*x_1^2 + y_1^2)
    X_train_sig = np.sqrt(np.sqrt(10*X_train_real[:, 2]**2
                          + X_train_real[:, 3]**2)) \
        - np.sqrt(np.sqrt(10*X_train_real[:, 0]**2 + X_train_real[:, 1]**2))
    X_train_real = X_train_real[X_train_sig > 0]
    # exclude training points where turbine 3 is more important
    # than turbine 2 using distance = sqrt(10*x_1^2 + y_1^2)
    X_train_sig = np.sqrt(np.sqrt(10*X_train_real[:, 4]**2
                          + X_train_real[:, 5]**2)) \
        - np.sqrt(np.sqrt(10*X_train_real[:, 2]**2 + X_train_real[:, 3]**2))
    X_train_real = X_train_real[X_train_sig > 0]
    # run simulations to find data points
    y_train = np.zeros(len(X_train_real))
    for i in range(len(X_train_real)):
        y_train[i] = simulator6d(X_train_real[i, :], noise_level)
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
        y_train[i] = simulator6d(X_train_real[i, :], noise_level)
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
        y_test[i] = simulator6d(X_test_real[i, :], noise_level)
    return X_test_real, y_test
