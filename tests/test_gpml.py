import numpy as np
import numpy.testing as npt
from GP_machine_learning.GP_machine_learning_functions import *
from regular_array_sampling.functions import *


def test_testing_data():
    """Confirm that the testing points are valid
    """
    X_test, y_test = create_testing_points(0.01)
    # test that turbine1 is at least 2D from origin
    X_test_dist = np.sqrt(X_test[:, 0]**2 + X_test[:, 1]**2)
    assert(np.all(X_test_dist > 2))
    # test that turbine2 is more important than 1
    X_test_sig = calculate_distance(X_test[:, 2],
                                    X_test[:, 3]) \
        - calculate_distance(X_test[:, 0], X_test[:, 1])
    assert(np.all(X_test_sig > 0))
    # test that turbine2 is more important than 1
    X_test_sig = calculate_distance(X_test[:, 4],
                                    X_test[:, 5]) \
        - calculate_distance(X_test[:, 2], X_test[:, 3])
    assert(np.all(X_test_sig > 0))
    # test that y1, y2, y3 is in the range [-5, 5]
    assert(np.all(np.abs(X_test[:, 1]) < 5))
    assert(np.all(np.abs(X_test[:, 3]) < 5))
    assert(np.all(np.abs(X_test[:, 5]) < 5))
    # test that x1, x2, x3 is in the range [0, 30]
    assert(np.all(X_test[:,0] < 30))
    assert(np.all(X_test[:,2] < 30))
    assert(np.all(X_test[:,4] < 30))
    assert(np.all(X_test[:,0] >= 0))
    assert(np.all(X_test[:,2] >= 0))
    assert(np.all(X_test[:,4] >= 0))

def test_regular_training_data():
    """Confirm that the training points are valid
    Training points are generated by sampling 
    regular turbine arrays
    """
    cand_points = regular_array_monte_carlo(100)
    X_test, y_test, n_train = create_training_points_regular(100, 0.01, cand_points)
    # test that turbine1 is at least 2D from origin
    X_test_dist = np.sqrt(X_test[:, 0]**2 + X_test[:, 1]**2)
    assert(np.all(X_test_dist >= 2))
    # test that turbine2 is more important than 1
    X_test_sig = calculate_distance(X_test[:, 2],
                                    X_test[:, 3]) \
        - calculate_distance(X_test[:, 0], X_test[:, 1])
    assert(np.all(X_test_sig > 0))
    # test that turbine2 is more important than 1
    X_test_sig = calculate_distance(X_test[:, 4],
                                    X_test[:, 5]) \
        - calculate_distance(X_test[:, 2], X_test[:, 3])
    assert(np.all(X_test_sig > 0))
    # test that y1, y2, y3 is in the range [-5, 5]
    assert(np.all(np.abs(X_test[:, 1]) < 5))
    assert(np.all(np.abs(X_test[:, 3]) < 5))
    assert(np.all(np.abs(X_test[:, 5]) < 5))
    # test that x1, x2, x3 is in the range [0, 30]
    assert(np.all(X_test[:,0] < 30))
    assert(np.all(X_test[:,2] < 30))
    assert(np.all(X_test[:,4] < 30))
    assert(np.all(X_test[:,0] >= 0))
    assert(np.all(X_test[:,2] >= 0))
    assert(np.all(X_test[:,4] >= 0))