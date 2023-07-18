import os
import glob
import re
import pandas as pd
import torch
import yaml
from autoSSL.models import pipe_model
from autoSSL.models.Backbone import pipe_backbone

def pipe_loadckpt(ckpt):

    checkpoint = torch.load(ckpt)
    # Extract only backbone state_dict
    state_dict_backbone = {k: v for k, v in checkpoint['state_dict'].items() if 'backbone' in k and 'momentum' not in k}
    # Remove 'backbone.' prefix in state_dict keys
    state_dict_backbone = {k.replace('backbone.', ''): v for k, v in state_dict_backbone.items()}
    #state_dict_backbone.keys()
    
    
    try:
        model_temp=pipe_backbone("resnet18_5layer")[0]
        model_temp.load_state_dict(state_dict_backbone)
        model=model_temp
        print("Successfully load resnet18_5layer")
    except:
        pass
    
    try:
        model_temp=pipe_backbone("resnet18_5layer_split8")[0]
        model_temp.load_state_dict(state_dict_backbone)
        model=model_temp
        print("Successfully load resnet18_5layer_split8")
    except:
        pass
    
    try:
        model_temp=pipe_backbone("resnet18")[0]
        model_temp.load_state_dict(state_dict_backbone)
        model=model_temp
        print("Successfully load resnet18")
    except:
        #raise ValueError("Cannot detect the current ckpt")
        pass

    try:
        model.eval()
        return model
    except:
         
        raise ValueError("Cannot detect the current ckpt")
        
def pipe_collate(address, reg, autoDL=None):
    # Define the directory to search
    search_dir = address
    
    # Use * as a wildcard to match all files and directories starting with "batch_"
    pattern = os.path.join(search_dir, "*")

    # Find all files and directories under the search directory
    matching_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

    # Filter directories based on the regular expression
    regex = re.compile(reg)
    matching_dirs = [d for d in matching_dirs if regex.search(os.path.basename(d))]
    
    # Initialize lists to store column data
    dir_names = []
    ckpt_paths = []
    config_paths = []
    log_paths = []
    
    model_list = []  # list to store loaded models
    
    # Loop through all matching directories
    for dir_path in matching_dirs:
        dir_name = os.path.basename(dir_path)
        dir_names.append(dir_name)

        # Find the checkpoint file with maximum epoch
        checkpoint_files = glob.glob(os.path.join(dir_path, "**", "*.ckpt"), recursive=True)

        max_epoch = -1
        ckpt_path = None
        for file in checkpoint_files:
            # extract epoch number from filename
            epoch_str = file.split('=')[1].split('-')[0]
            epoch = int(epoch_str)
            if epoch > max_epoch:
                max_epoch = epoch
                ckpt_path = file
        ckpt_paths.append(ckpt_path)

        # Generate the config file path and log file path
        config_path = os.path.join(dir_path, "config.yaml")
        config_paths.append(config_path)
        log_paths.append(os.path.join(dir_path, dir_name+".csv"))

        # Load model and add to model_list
        if autoDL:
            print(ckpt_path)
            model=pipe_loadckpt(ckpt_path)
        else:
            try:
                checkpoint = torch.load(ckpt_path)
            except:
                print(dir_path)
                print(ckpt_path)
                raise("error")
            with open(config_path, 'r') as stream:
                config = yaml.safe_load(stream)

            model = pipe_model(config=config) 
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            model=model.backbone
        model_list.append(model)
    
    # Create a pandas DataFrame
    df = pd.DataFrame({
        'dir_name': dir_names,
        'ckpt_path': ckpt_paths,
        'config_path': config_paths,
        'log_path': log_paths
    })
    invalid_chars = '[<>:"/\|?*]'
    # Replace each invalid character with '_'
    safe_reg = re.sub(invalid_chars, '_', reg)

    # Use the cleaned string to create your file path
    csv_path = os.path.join(search_dir, safe_reg + ".csv")
    
    print(f"Collating the models' (evaluating) information to {csv_path}")
    
    df.to_csv(csv_path, index=False)
    
    return {'name': dir_names, 'model': model_list, 'address': csv_path}

