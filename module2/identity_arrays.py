"""Practice some of numpy array identities
"""
import numpy as np
def main():
    """Driven Function
    """
    # Create a 2D 3x3 identity matrix
    identity_3x3 = np.eye(3,3)
    print (identity_3x3)
    identity_3x5 = np.eye(3,5)
    print (identity_3x5)
    # 2D Diagonal arrays, with given entries
    diagonal_2D = np.diag([2,3,4,5])
    print(diagonal_2D)
    

if(__name__ == '__main__'):
    main()
