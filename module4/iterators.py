""" Iterator protocols
"""
import numpy as np
def main():
    """Driven Function
    """
    iterable = ['Freshman', 'Sophomore', 'Junior', 'Senior']
    # Create an Iterator
    iterator = iter(iterable)
    # Print the next item from the iterator
    print(next(iterator))
    print(next(iterator))
    print(next(iterator))
    print(next(iterator))
    # print(next(iterator)) # StopIteration Error
    # TODO: Add a function with a try: catch: to test the iterator
    
    # TODO: Then, use a Generator
if(__name__ == '__main__'):
    main()
