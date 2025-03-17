import os
import datetime

from pathlib import Path

from src.utils.file_utils import check_url_exists, fetch_url, unzip, set_github_output
from src.utils.file_io import save_file, load_file


def execute():

    config = load_file('config.yaml')
    DATA_DIR = Path(config['paths']['data'])
    VERSION_PATH =  DATA_DIR / config['files']['version']
    RAW_DATA_PATH =  DATA_DIR /config['files']['raw_data']
    SOURCE_URL = config['source_url']
    MAX_LOOKBACK_MONTHS = config['max_lookback_months']

    # date-year stamps for versions
    latest_version = get_latest(SOURCE_URL, MAX_LOOKBACK_MONTHS)
    try:
        current_version = load_file(VERSION_PATH)
    except:
        current_version = None

    if current_version == latest_version:
        return  # Exit early if the data is up-to-date.

    # download updated data 
    update_data(latest_version, SOURCE_URL, DATA_DIR, RAW_DATA_PATH)

    # update version file
    data_version = {'version': f"{latest_version[0]}-{latest_version[1]}"}
    save_file(data_version, VERSION_PATH)
        
    # If running in GitHub Actions, set output    
    if "GITHUB_OUTPUT" in os.environ :
        set_github_output('status', 'updated')
   

def get_latest(url, lookback_months):
    """ Searches StatCan for the most recent LFS microdata file"""
    current_date = datetime.datetime.now()
    month = current_date.month
    year = current_date.year

    for _ in range(lookback_months):
        year, month = (year, month-1) if month > 1 else (year-1, 12)
        file_name = f"{year}-{month:02d}-CSV.zip"
        if check_url_exists(url + file_name):
            return [month, year]
    
    raise Exception(f"No LFS data found at source in past {lookback_months} months")


def update_data(date, source_url, data_directory, raw_data_path):
    """Download and unzip the most recent data file."""

    # Create data directory if it doesn't exist
    data_directory.mkdir(parents=True, exist_ok=True)

    # Clear existing data if applicable
    for item in data_directory.iterdir():
        if item.is_file():
            item.unlink()

    file_name = f"{date[1]}-{date[0]:02d}-CSV.zip"
    file_url = source_url + file_name
    file_path = data_directory / file_name

    #download + unzip
    fetch_url(file_url, file_path)
    unzip(file_path)
    
    #rename
    csv_path = data_directory / f"pub{date[0]:02d}{str(date[1])[-2:]}.csv"
    Path(csv_path).rename(raw_data_path)

    
if __name__ == "__main__":
    execute()
