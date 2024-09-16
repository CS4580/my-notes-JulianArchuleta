"""Download data from our server
"""
import requests
import zipp
SERVER_URL = 'http://icarus.cs.weber.edu/~hvalle/cs4580/data/'

def download_file(url, file_name):
    # TODO: Download to pwd
    file = requests.get(url)   
    print(f'Error: {file.status_code}')
    # TODO: Check extension, if it zip call unzip_file
    if file_name.endswith('.zip'):
        unzip_file(file)
    pass


def unzip_file(file_name):
    # TODO: Unzip file
    with zipp.zipfile.ZipFile(file_name) as file:
        file.extractall()
    pass


def main():
    """Driven Function
    """
    data01 = 'pandas01Data.zip'
    file = download_file(SERVER_URL,  data01)
    
    
    

if(__name__ == '__main__'):
    main()
