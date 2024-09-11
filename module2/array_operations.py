"""Do array operations
"""
import numpy as np
def main():
    """Driven Function
    """
    num_list = [2,4,6,8,10]
    print(f'Before{num_list}')
    for num in range(len(num_list)):
        num_list[num] = num_list[num] + 3
    print(f'After{num_list}')
    # Convert list to an Numpy array
    num_arr = np.array(num_list)
    num_arr += 3
    print(f'Numpy Array{num_arr}')
        
if(__name__ == '__main__'):
    main()
