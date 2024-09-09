"""Practice creating arrays from ranges
"""

import numpy as np

def main():
    """Driven Function
    """

    # Generate a 1D array with values from 0 to 8
    arr_1d = np.arange(9)
    print(arr_1d)
    # You may have negative values
    array = np.arange(-4,4)
    print(array)
    # Add a step to each increment
    array = np.arange(-8,8,2)
    print(array)

    # Generate an array with values 0 to 5, in steps of 0.1
    array = np.arange(0,5,0.1)
    print(array)
    # Ranfges with decimal values
    array = np.arange(0.1, 0.3, 0.01)
    print(array)
if(__name__ == '__main__'):
    main()
