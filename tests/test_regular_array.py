import numpy as np
import numpy.testing as npt
from regular_array_sampling.functions import find_important_turbines


def test_closest_turbines_calc():
    """Test that find_important_turbines correctly calculates
    the coordinates of the closest turbines
    """
    test_answer = np.array([[0, 6], [0, 12], [0, 18]])
    npt.assert_array_equal(find_important_turbines(6,6,0,0), test_answer)
    npt.assert_array_almost_equal(find_important_turbines(6,6,0,np.pi/2), test_answer)
    test_answer = np.array([[1.5, 6], [3, 12], [-6, 12]])
    npt.assert_array_equal(find_important_turbines(9,6,1.5,0), test_answer)
    test_answer = np.array([[0, 3], [0, 6], [0, 9]])
    npt.assert_array_almost_equal(find_important_turbines(3,6,0,np.pi/2), test_answer)
    test_answer = np.array([-6*np.sin(0.1), 6*np.cos(0.1)])
    npt.assert_array_almost_equal(find_important_turbines(6,6,0,0.1)[0,:], test_answer)