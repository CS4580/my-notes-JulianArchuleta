"""Practice some of numpy array identities
"""
import numpy as np
def main():
    """Driven Function
    """
    # Create a 2D 3x3 identity matrix
    identity_3x3 = np.eye(3,3)
    print (f'identity_3x3\n{identity_3x3}')
    identity_3x5 = np.eye(3,5)
    print (f'identity_3x5\n{identity_3x5}')
    # 2D Diagonal arrays, with given entries
    diagonal_2D = np.diag([2,3,4,5])
    print(f'diagonal_2D\n{diagonal_2D}')
    
    #Create a 5x3 2D array of unsigned intergers filled with zeros
    zeros_5x3 = np.zeros((5,3), dtype=np.uint)
    print(f'zeros_5x3\n{zeros_5x3}')
    # Create a 5x3 2D array of unsigned intergers filled with ones
    ones_5x3 = np.ones((5,3), dtype=np.uint)
    print(f'ones_5x3\n{ones_5x3}')
    # Create a 5x3 2D array of unsigned intergers filled with pi
    pi_5x3 = np.full((5,3), np.pi)
    print(f'pi_5x3\n{pi_5x3}')
    # Create a 5x3 2D array of unsigned intergers filled with random values
    random_5x3 = np.random.random((5,3))
    print(f'random_5x3\n{random_5x3}')


    

if(__name__ == '__main__'):
    main()
