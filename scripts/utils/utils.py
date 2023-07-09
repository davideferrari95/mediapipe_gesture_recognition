import os, shutil
import rospy
import torch

# Get Torch Device ('cuda' or 'cpu')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Countdown Function
def countdown(num_of_secs):

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
    yaml = ruamel.yaml.YAML()

    hydra_config_file = os.path.join(FOLDER, 'config/config.yaml')

    # Load Hydra `config.yaml` File
    with open(hydra_config_file, 'r') as file:
        yaml_data = yaml.load(file)

    # Edit Hydra Run Directory
    yaml_data['hydra']['run']['dir'] = os.path.join(FOLDER, r'data/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}')

    # Write Hydra `config.yaml` File
    with open(hydra_config_file, 'w') as file:
        yaml.dump(yaml_data, file)

def delete_pycache_folders():

    """ Delete Python `__pycache__` Folders Function """

    # Walk Through the Project Folders
    for root, dirs, files in os.walk(FOLDER):

        if "__pycache__" in dirs:

            # Get `__pycache__` Path
            pycache_folder = os.path.join(root, "__pycache__")
            print(f"Deleting {pycache_folder}")

            # Delete `__pycache__`
            try: shutil.rmtree(pycache_folder)
            except Exception as e: print(f"An error occurred while deleting {pycache_folder}: {e}")

def handle_signal(signal, frame):

    # SIGINT (Ctrl+C)
    print("\nProgram Interrupted. Deleting `__pycache__` Folders...")
    delete_pycache_folders()
    print("Done\n")
    exit(0)
