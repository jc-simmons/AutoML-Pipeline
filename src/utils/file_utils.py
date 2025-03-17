import os
from zipfile import ZipFile
import requests
import urllib.request
import os


def fetch_url(file_url, file_name):
    """ Downloads a file from the specified URL and saves it with the given file name. """
    urllib.request.urlretrieve(file_url, file_name)
    return


def check_url_exists(path):
    """ Checks if a file exists at the given URL by sending a HEAD request. """
    try:
        response = requests.head(path)
        return response.status_code == requests.codes.ok
    except requests.RequestException as e:
        print(f"Error accessing url file: {e}")
        return False
    
    
def unzip(zip_file):
    """ Extracts the contents of the specified ZIP file and removes the archive. """
    zip_dir = os.path.dirname(zip_file)

    with ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(zip_dir)
    os.remove(zip_file)
    return 


def set_github_output(output_name, value):
    """Sets a GitHub Action output variable."""
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"{output_name}={value}\n")