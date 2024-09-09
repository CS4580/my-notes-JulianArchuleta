"""1D Array
"""
import numpy as np

def main():
    """Driven Function
    """
    # Create an array
    array = np.array([-2, 1, -5, 10])

    print(array, type(array))
    numbers = [-2, 1, -5, 10]
    print(numbers, type(numbers))
    # Convert list to array
    new_array = np.array(numbers)
    print(new_array, type(new_array))

    # 2D Arrays
    matrix = np.array([[-1, 0,4], 
                       [-3,6,9]])
    print(f'2D array\n{matrix}')

    # 3D Arrays
    array3d = np.array([[[1, 2, 3], 
                         [4, 5, 6]], 
                         [[7, 8, 9], 
                          [10, 11, 12]]
                          ])
    print(f'3D array\n{array3d}')
    
    # Use the dtype optional argument to explicitly
    # call the data type of the array
    numbers = [-2, 1, -5, 10]
    new_array = np.array(numbers, dtype=np.short)
    print(new_array, new_array.dtype)

    pos_numbers = [2, 1,5, 10]
    new_array = np.array(pos_numbers, dtype=np.ushort)
    print(new_array, new_array.dtype)
    



    

if(__name__ == '__main__'):
    main()
