import yaml
import os

def load_config(origin_config, model_name):
    # Copy the original configuration
    new_config = origin_config.copy()
    
    # Construct the path to the YAML file
    yaml_file = model_name + ".yaml"
    
    # Check if the YAML file exists
    if not os.path.isfile(yaml_file):
        raise FileNotFoundError(f"No YAML configuration file found at {yaml_file}")
    
    # Load the YAML configuration
    with open(yaml_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Overwrite items in the new configuration
    for item, value in yaml_config.items():
        if item not in new_config:
            raise KeyError(f"Item {item} not found in original configuration")
        new_config[item] = value
    
    return new_config
