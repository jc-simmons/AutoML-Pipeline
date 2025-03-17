import itertools
import importlib

def dict_combinations(input_dict: dict): 
    """ Generates all possible combinations of key-value pairs from the input dictionary of lists. """
    keys = list(input_dict.keys())
    values = list(input_dict.values())
    combinations = itertools.product(*values)
    result = []
    for combo in combinations:
        result.append(dict(zip(keys, combo))) 
    return result


def dyanmic_loader(entity):
    """ Loads and returns an instantiated entity with optional kwarg dictionary. """
    construct, kargs = entity
    c_init = import_construct(construct)
    res = c_init(**kargs)
    return res

def import_construct(full_construct_path: str):
    """ Dynamically import a construct from a module given the full path."""
    module_name, class_name = full_construct_path.rsplit('.', 1)  
    module = importlib.import_module(module_name) 
    class_obj = getattr(module, class_name)  
    return class_obj  


def generate_model_combinations(model_config_dict):
    """ Generates a list of models instantiated with all combinations of parameters from the input dictionary.
    The resulting list contains tuples of the model instance and a dictionary of the model info. """
    model_instances_with_params = [
        (dyanmic_loader((model, params)), params) 
        for model in model_config_dict 
        for params in dict_combinations(model_config_dict[model])
    ]
    model_with_metadata = [
        (model, {'model': model.__class__.__name__, 'params': params}) 
        for model, params in model_instances_with_params
    ]
    
    return model_with_metadata