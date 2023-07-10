import os, shutil, yaml, torch
from pytorch_lightning import LightningModule

# Get Torch Device ('cuda' or 'cpu')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Countdown Function
def countdown(num_of_secs):

  import rospy

  print("\nAcquisition Starts in:")

  # Wait Until 0 Seconds Remaining
  while (not rospy.is_shutdown() and num_of_secs != 0):
    m, s = divmod(num_of_secs, 60)
    min_sec_format = '{:02d}:{:02d}'.format(m, s)
    print(min_sec_format)
    rospy.sleep(1)
    num_of_secs -= 1

  print("\nSTART\n")

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

def save_parameters(path, file_name, **kwargs):

  """ Save Parameters Function """

  # Create Directory if it Doesn't Exist
  os.makedirs(path, exist_ok=True)

  with open(os.path.join(path, file_name), 'w') as file:
    yaml.dump(kwargs, file, sort_keys=False)

def load_parameters(file_name) -> dict:

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

from pytorch_lightning.callbacks import Callback
from termcolor import colored

# Print Start Training Info Callback
class StartTrainingCallback(Callback):

  # On Start Training
  def on_train_start(self, trainer, pl_module):
    print(colored('\n\nStart Training Process\n\n','yellow'))

  # On End Training
  def on_train_end(self, trainer, pl_module):
    print(colored('\n\nTraining Done\n\n','yellow'))

# Print Start Validation Info Callback
class StartValidationCallback(Callback):

  # On Start Validation
  def on_validation_start(self, trainer, pl_module):
    print(colored('\n\n\nStart Validation Process\n\n','yellow'))

  # On End Validation
  def on_validation_end(self, trainer, pl_module):
    print(colored('\n\n\n\nValidation Done\n\n','yellow'))

# Print Start Testing Info Callback
class StartTestingCallback(Callback):

  # On Start Testing
  def on_test_start(self, trainer, pl_module):
    print(colored('\n\nStart Testing Process\n\n','yellow'))

  # On End Testing
  def on_test_end(self, trainer, pl_module):
    print(colored('\n\n\nTesting Done\n\n','yellow'))
