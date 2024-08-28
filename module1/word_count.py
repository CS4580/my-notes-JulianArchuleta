"""Read file from web and do analysis of data
"""
from urllib.request import urlopen

def count_words_from_web_file(file_address):    
    words = 0
    #TODO: Read file from web
    with urlopen(file_address) as data:
        #TODO: Count number of words
        words = len(data.read().decode().split())
    return words
    



def main():
    """Driven Function
    """
    file_address = 'http://icarus.cs.weber.edu/~hvalle/sample_data/poem.txt'
    words = count_words_from_web_file(file_address)
    print(f'There are {words} words in {file_address}')

if(__name__ == '__main__'):
    main()
