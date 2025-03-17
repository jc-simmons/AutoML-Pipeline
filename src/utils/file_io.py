from pathlib import Path


def get_extension(file_name):
    """Get the file extension from the file name."""
    file_extension = file_name.suffix.lower()
    return file_extension

def get_io_function(file_name):
    """Retrieve the appropriate handle function for the file extension."""
    extension = get_extension(file_name)
    
    supported_extensions = {
        ".json": handle_json,
        ".yaml": handle_yaml,
        ".csv": handle_csv,
        ".pkl": handle_pkl,
        ".png": handle_png,
        ".joblib": handle_joblib
    }
    
    try:
        return supported_extensions[extension]
    except KeyError:
        raise ValueError(f"Unsupported extension '{extension}' ")

def perform_io(mode, file_name, **kwargs):
    """ Helper method to perform I/O operations (load/save). """
    data = kwargs.pop('data', None)
    # Ensure file_name is a Path object (whether string or Path was passed)
    file_name = Path(file_name) if not isinstance(file_name, Path) else file_name
    
    io_function = get_io_function(file_name)
    return io_function(file_name, mode, data, **kwargs)

def load_file(file_name, **kwargs):
    """ Load data from the file. """
    return perform_io('r', file_name, **kwargs)

def save_file(data, file_name, **kwargs):
    """ Save data to file. """
    perform_io('w', file_name, data = data, **kwargs)


def handle_json(file_name, mode, data=None, **kwargs):
    """Handle read/write for JSON files."""
    import json  # Import only when needed

    if mode == 'r':
        with open(file_name, 'r') as f:
            return json.load(f, **kwargs)
    elif mode == 'w':
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)

def handle_yaml(file_name, mode, data=None, **kwargs):
    """Handle read/write for YAML files."""
    import yaml  # Import only when needed

    if mode == 'r':
        with open(file_name, 'r') as f:
            return yaml.safe_load(f)
    elif mode == 'w':
        with open(file_name, 'w') as f:
            yaml.dump(data, f)

def handle_csv(file_name, mode, data=None, **kwargs):
    """Handle read/write for CSV files."""
    import pandas as pd  # Import only when needed

    if mode == 'r':
        return pd.read_csv(file_name, **kwargs)
    elif mode == 'w':
        data.to_csv(file_name, **kwargs)


def handle_joblib(file_name, mode, data=None, **kwargs):
    """Handle read/write for Joblib files."""
    import joblib  # Import only when needed

    if mode == 'r':
        return joblib.load(file_name, **kwargs)
    elif mode == 'w':
        joblib.dump(data, file_name, **kwargs)


def handle_pkl(file_name, mode, data=None, **kwargs):
    """Handle read/write for Pickle files."""
    import pickle  # Import only when needed
    default_kwargs = {'protocol': pickle.HIGHEST_PROTOCOL}
    kwargs = {**default_kwargs, **kwargs}
    
    if mode == 'r':
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    elif mode == 'w':
        with open(file_name, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def handle_png(file_name, mode, fig = None, **kwargs):
    """ Utility function to save a matplotlib figure with customizable options """
    import matplotlib.pyplot as plt
    default_kwargs = {'dpi':1200, 'bbox_inches':'tight'}
    kwargs = {**default_kwargs, **kwargs} 
    plt.figure(fig)
    plt.savefig(file_name, **kwargs)


