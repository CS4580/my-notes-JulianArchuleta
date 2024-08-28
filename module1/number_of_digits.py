"""Library to calculate the number of digits for different algorithms
"""
from math import factorial

def factorial_length(number):
    """Count the number of digits in a factorial

    Args:
        number (int): integer value to calculate factorial

    Returns:
        int: number of digits in the factorial of input
    """  
    return len(str(factorial(number)))  # convert the factorial to string then count the number of elements



def main():
    """Driven Function
    """
    num = 5
    digits = factorial_length(num)
    print (f"You have {digits} digits in the factorial of {num}")

if(__name__ == '__main__'):
    main()
