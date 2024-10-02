"""Download data from our server
"""
import zipfile
import os
import requests
from io import BytesIO
SERVER_URL = "http://icarus.cs.weber.edu/~hvalle/cs4580/data/"
FILE_NAME = "plottingDemo01.zip"


def download_file(url, file_name):
    """Download a file from the server and save it to pwd
    
    Parameters
    ----------
    url : str
        The URL of the file to download
    file_name : str
        The name of the file to save it to
    """
    # TODO: Download to pwd
    url = url + file_name
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, 'wb') as file:
            file.write(response.content)
        print(f'Downloaded {file_name}')
        if zipfile.is_zipfile(file_name):
            unzip_file(file_name)
    else:
        print(f'Download failed: {response.status_code}')
    


def unzip_file(file_name):
    """Unzip a file in the current working directory
    
    Parameters
    ----------
    file_name : str
        Name of the file to be unzipped
    """
    # TODO: Unzip file    
    with zipfile.ZipFile(file_name, "r") as zip_ref:
        zip_ref.extractall("data")
        print(f'Unzipped {file_name}')
    os.remove(file_name)
    
    


def main(server_url, file_name):
    """Driven Function
    """    
    download_file(server_url,  file_name)
    #unzip_file(file_name)
    

if(__name__ == '__main__'):
    main(SERVER_URL, FILE_NAME)
