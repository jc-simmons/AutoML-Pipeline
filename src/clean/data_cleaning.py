import warnings
import functools

from pathlib import Path
import pandas as pd
from collections import defaultdict

from src.utils.file_io import load_file, save_file

def execute():

    config = load_file('config.yaml')
    DATA_DIR = Path(config['paths']['data'])
    RAW_DATA_PATH  = DATA_DIR / config['files']['raw_data']
    CLEANED_DATA_PATH = DATA_DIR / config['files']['cleaned_data']
    CODEBOOK_PATH = DATA_DIR / config['files']['codebook']
    DECODE_COLUMN_MAP =  config['decode_column_map']

    warnings.simplefilter(action='ignore', category=FutureWarning)

    raw_data = load_file(RAW_DATA_PATH)
    # preserve numerical age ordering for later model use
    age_12_numerical_labels = raw_data['AGE_12'].copy()

    # decode variables based on LFS_PUMF_EPA_FGMD_codebook 
    code_dict = generate_decode_dict(CODEBOOK_PATH, DECODE_COLUMN_MAP) 
    decoded_data = decode_lfs_labels(raw_data, code_dict) 

    # keep 2 AGE_12 columns: ordered numerical is useful for modelling, labelled categorical for easy analysis
    decoded_data = decoded_data.assign(AGE_12_NUM = age_12_numerical_labels)
    decoded_data.rename(columns={'AGE_12': 'AGE_12_CAT'}, inplace=True)

    # run several filtering operations + save result
    cleaned_data = clean_data(decoded_data)
    save_file(cleaned_data, CLEANED_DATA_PATH, index = False)

def generate_decode_dict(codebook_path, decode_column_map):
    """Parses the codebook to generate a decoding dictionary"""
    code_list=pd.read_csv(codebook_path, encoding = 'latin1')
    code_dict = defaultdict(dict)

    for _, row in code_list.iterrows():

        field_num = row[decode_column_map['field_num']]

        if not pd.isna(field_num):  # Start of a new variable
            variable = row[decode_column_map['variable']].upper().strip()
        else:  # Variable value and label
            try:
                variable_val = int(row[decode_column_map['variable']]) 
            except ValueError:
                variable_val = row[decode_column_map['variable']] 

            label = row[decode_column_map['label']]
            code_dict[variable][variable_val] = label

    return code_dict


def decode_lfs_labels(df, code_dict):
    """ Decodes labels based on the code dictionary."""
    for variable in df.columns:
        for variable_val in code_dict[variable]:
            if (type(variable_val)==int):
                df.loc[df[variable] == variable_val,variable] = code_dict[variable][variable_val]

    return df


def func_pipeline(*functions):
    """" Function composer to avoid storing intermediate stages."""
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)

def clean_data(df):
    """Cleans the dataset by removing unnecessary data and applying filters."""
    clean_pipeline = func_pipeline(drop_unemployment, drop_cardinal, rescale_income, filter_valid_wages)
    return clean_pipeline(df)


def filter_valid_wages(df):
    """Filters rows based on valid minimum wages by province."""
    min_wage = {
        'Alberta': 15.00, 'British Columbia': 16.75, 'Manitoba': 15.30,
        'New Brunswick': 14.75, 'Newfoundland and Labrador': 15.00, 'Nova Scotia': 15.00,
        'Ontario': 16.55, 'Prince Edward Island': 15.00, 'Quebec': 15.25, 'Saskatchewan': 14.00
    }
    # Check if wage is above minimum for each province
    valid_wage = df['HRLYEARN'] > df['PROV'].map(min_wage)
    df = df[valid_wage]

    return df


def drop_unemployment(df):
    """Series of steps to drop rows and columns indicating an unemployed status."""
    # Remove rows with no reported income
    df = df.dropna(subset=['HRLYEARN'])

    # Only keep rows where the person is "Employed, at work"
    df = df[df['LFSSTAT'] == 'Employed, at work']

    #columns indicating unemployment
    unemployed_cols = [
        'DURUNEMP', 'FLOWUNEM', 'UNEMFTPT', 'WHYLEFTO', 'WHYLEFTN', 'DURJLESS',
        'AVAILABL', 'LKEMPLOY', 'LKRELS', 'LKATADS', 'LKANSADS', 'LKOTHERN',
        'PRIORACT', 'YNOLOOK', 'TLOLOOK', 'LKPUBAG', 'EVERWORK', 'PREVTEN', 'FTPTLAST'
    ]

    # Drop rows with any non-NaN values in unemployment columns
    df = df[df[unemployed_cols].isna().all(axis=1)]

    # Drop unemployment-related columns and labour status 
    df = df.drop(columns=unemployed_cols + ['LFSSTAT'])

    return df


def drop_cardinal(df):
    """Drops cardinal data not relevant for analysis or modeling."""
    cardinal_cols = ['REC_NUM' , 'FINALWT']
    df = df.drop(columns=cardinal_cols)
    return df


def rescale_income(df):
    """Rescale income from cents to dollars."""
    df['HRLYEARN'] = df['HRLYEARN'] / 100.0
    return df


if __name__ == "__main__":
    execute()
