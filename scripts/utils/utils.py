import os, shutil, yaml, torch
from pytorch_lightning import LightningModule
from termcolor import colored

# Get Torch Device ('cuda' or 'cpu')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Project Folder (ROOT Project Location)
FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))

def set_hydra_absolute_path():

    import ruamel.yaml
    ruamel_yaml = ruamel.yaml.YAML()

    hydra_config_file = os.path.join(FOLDER, 'config/config.yaml')

    # Load Hydra `config.yaml` File
    with open(hydra_config_file, 'r') as file:
        yaml_data = ruamel_yaml.load(file)

    # Edit Hydra Run Directory
    yaml_data['hydra']['run']['dir'] = os.path.join(FOLDER, r'data/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}')

    # Write Hydra `config.yaml` File
    with open(hydra_config_file, 'w') as file:
        ruamel_yaml.dump(yaml_data, file)

def save_parameters(path, file_name:str, **kwargs):

    """ Save Parameters Function """

    # Create Directory if it Doesn't Exist
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, file_name), 'w') as file:
        yaml.dump(kwargs, file, sort_keys=False)

def load_parameters(file_name:str) -> dict:

    """ Load Parameters Function """

    with open(file_name, 'r') as file:
        parameters = yaml.safe_load(file)

    return parameters

def save_model(path:str, file_name:str, model:LightningModule):

    """ Save File Function """

    # Create Directory if it Doesn't Exist
    os.makedirs(path, exist_ok=True)
    counter = 1

    # Check if the File Already Exists
    while os.path.exists(os.path.join(path, file_name)):

        # Get the File Name and Extension
        file_name, file_extension = os.path.splitext(file_name)

        # Append a Number to the File Name to Make it Unique
        counter += 1
        file_name = f'{file_name}_{counter}{file_extension}'
        print(f'Model Name Already Exists. Saving as {file_name}')

    with open(os.path.join(path, file_name), 'wb') as FILE: torch.save(model.state_dict(), FILE)
    print(colored('\n\nModel Saved Correctly\n\n', 'green'))

def delete_pycache_folders(verbose:bool=False):

    """ Delete Python `__pycache__` Folders Function """

    # Walk Through the Project Folders
    for root, dirs, files in os.walk(FOLDER):

        if "__pycache__" in dirs:

            # Get `__pycache__` Path
            pycache_folder = os.path.join(root, "__pycache__")
            if verbose: print(f"Deleting {pycache_folder}")

            # Delete `__pycache__`
            try: shutil.rmtree(pycache_folder)
            except Exception as e: print(f"An error occurred while deleting {pycache_folder}: {e}")

    if verbose: print('\n\n')

def handle_signal(signal, frame):

    # SIGINT (Ctrl+C)
    print("\nProgram Interrupted. Deleting `__pycache__` Folders...")
    delete_pycache_folders()
    print("Done\n")
    exit(0)
