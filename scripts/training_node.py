#!/usr/bin/env python3

import sys, os, signal, logging
import pickle, numpy as np
from typing import Union, Tuple
from termcolor import colored

# Import PyTorch Lightning
import torch, torch.nn as nn, torch.nn.functional as F
from pytorch_lightning import Trainer, LightningModule, loggers as pl_loggers
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import Dataset, DataLoader, random_split
from utils.utils import StartTrainingCallback, StartTestingCallback
from utils.utils import set_hydra_absolute_path, save_parameters, save_model

# Ignore Torch Compiler INFO
logging.getLogger('torch._dynamo').setLevel(logging.ERROR)
logging.getLogger('torch._inductor').setLevel(logging.ERROR)

# Set Torch Matmul Precision
torch.set_float32_matmul_precision('high')

# Import Signal Handler Function
from utils.utils import FOLDER, DEVICE, handle_signal, delete_pycache_folders
signal.signal(signal.SIGINT, handle_signal)

# Import Parent Folders
sys.path.append(FOLDER)

# Import Hydra and Parameters Configuration File
import hydra
from config.config import Params

# Set Hydra Absolute FilePath in `config.yaml`
set_hydra_absolute_path()

# Set Hydra Full Log Error
os.environ['HYDRA_FULL_ERROR'] = '1'

# Hydra Decorator to Load Configuration Files
@hydra.main(config_path=f'{FOLDER}/config', config_name='config', version_base=None)
def main(cfg: Params):

    # Create Gesture Recognition Training
    GRT = GestureRecognitionTraining3D(cfg)
    model, model_path = GRT.getModel()

    # Prepare Dataset
    train_dataloader, val_dataloader, test_dataloader = GRT.getDataloaders()

    # Create Trainer Module
    trainer = Trainer(

        # Devices
        devices = 'auto',
        accelerator = 'auto',

        # Hyperparameters
        min_epochs = cfg.min_epochs,
        max_epochs = cfg.max_epochs,
        log_every_n_steps = 1,

        # Instantiate Early Stopping Callback
        callbacks = [StartTrainingCallback(), StartTestingCallback(),
                    EarlyStopping(monitor='train_loss', mode='min', min_delta=cfg.min_delta, patience=cfg.patience, verbose=True)],

        # Use Python Profiler
        profiler = SimpleProfiler() if cfg.profiler else None,

        # Custom TensorBoard Logger
        logger = pl_loggers.TensorBoardLogger(save_dir=f'{FOLDER}/data/logs/'),

        # Developer Test Mode
        fast_dev_run = cfg.fast_dev_run

    )

    # Model Compilation
    compiled_model = torch.compile(model, mode=cfg.compilation_mode) if cfg.torch_compilation else model

    # Start Training
    trainer.fit(compiled_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(compiled_model, dataloaders=test_dataloader)

    # Save Model
    save_model(model_path, 'model.pth', compiled_model)

    # Delete Cache Folders
    delete_pycache_folders()

class GestureDataset(Dataset):

    """ Gesture Dataset Class """

    def __init__(self, x:Union[torch.Tensor, np.ndarray], y:Union[torch.Tensor, np.ndarray]):

        # Convert x,y to Torch Tensors
        self.x: torch.Tensor = x if torch.is_tensor(x) else torch.from_numpy(x).float()
        self.y: torch.Tensor = y if torch.is_tensor(y) else torch.from_numpy(y)

        # Convert Labels to Categorical Matrix (One-Hot Encoding)
        self.y: torch.Tensor = torch.nn.functional.one_hot(self.y)

        # Move to GPU
        self.x.to(DEVICE)
        self.y.to(DEVICE)

    def getInputShape(self) -> torch.Size:
        return self.x.shape

    def getOutputShape(self) -> torch.Size:
        return self.y.shape

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

class NeuralClassifier(LightningModule):

    """ Classifier Neural Network """

    def __init__(self, input_shape, output_shape, optimizer='Adam', lr=0.0005, loss_function='cross_entropy'):

        super(NeuralClassifier, self).__init__()

        # print(f'\n\nInput Shape: {input_shape} | Output Shape: {output_shape}')
        # print(f'Optimizer: {optimizer} | Learning Rate: {lr} | Loss Function: {loss_function}\n\n')

        # Compute Input and Output Sizes
        self.input_size, self.output_size = input_shape[1], output_shape[0]
        self.hidden_size, self.num_layers = 512, 1

        # Create LSTM Layers (Input Shape = Number of Flattened Keypoints (300 / 1734))
        # self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
        #                     batch_first=True, bidirectional=False).to(DEVICE)
        #                     # batch_first=True, bidirectional=False, dropout=0.5).to(DEVICE)

        # # Create Fully Connected Layers
        # self.fc_layers = nn.Sequential(
        #   nn.ReLU(),
        #   nn.Linear(self.hidden_size, 256),
        #   nn.ReLU(),
        #   # nn.Dropout(0.5),
        #   nn.Linear(256, 128),
        #   nn.ReLU(),
        #   nn.Linear(128, self.output_size)
        # ).to(DEVICE)

        # Create LSTM Layers -> Alberto Version
        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, self.output_size)

        # Instantiate Loss Function and Optimizer
        self.loss_function = getattr(torch.nn.functional, loss_function)
        self.optimizer     = getattr(torch.optim, optimizer)
        self.learning_rate = lr

    # def forward(self, x:torch.Tensor) -> torch.Tensor:

    #     # Hidden State and Internal State
    #     h_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(DEVICE)
    #     c_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(DEVICE)

    #     # Propagate Input through LSTM
    #     output, (hn, cn) = self.lstm(x, (h_0, c_0))

    #     # Reshaping Data for the Fully Connected Layers
    #     hn = hn.view(-1, self.hidden_size)

    #     # Forward Pass through Fully Connected Layers
    #     out = self.fc_layers(hn)

    #     # Softmax for Classification
    #     out = F.softmax(out, dim=1)

    #     return out

    def forward(self, x):

        x, _ = self.lstm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        x, _ = self.lstm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        x, _ = self.lstm3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        # Select only the last output from the LSTM
        x = x[:, -1, :]

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        x = self.fc2(x)

        return x

    def configure_optimizers(self):

        # Return Optimizer
        return self.optimizer(self.parameters(), lr = self.learning_rate)

    def compute_loss(self, batch:Tuple[torch.Tensor, torch.Tensor], log_name:str) -> torch.Tensor:

        # Get X,Y from Batch
        x, y = batch

        # Forward Pass
        y_pred = self(x)

        # Compute Loss
        loss = self.loss_function(y_pred, y.float())
        self.log(log_name, loss)

        # TODO: Compute Accuracy
        # acc = self.accuracy(y_pred, y)
        # self.log("train_acc", acc, prog_bar= True, on_step=True, on_epoch=False)

        return loss

    def training_step(self, batch, batch_idx):

        loss = self.compute_loss(batch, 'train_loss')
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        loss = self.compute_loss(batch, 'val_loss')
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):

        loss = self.compute_loss(batch, 'test_loss')
        return {'test_loss': loss}

class GestureRecognitionTraining3D:

    """ 3D Gesture Recognition Training Class """

    def __init__(self, cfg:Params):

        # Choose Gesture File
        gesture_file = ''
        if cfg.enable_right_hand: gesture_file += 'Right'
        if cfg.enable_left_hand:  gesture_file += 'Left'
        if cfg.enable_pose:       gesture_file += 'Pose'
        if cfg.enable_face:       gesture_file += 'Face'
        print(colored(f'\n\nLoading: {gesture_file} Configuration', 'yellow'))

        # Get Database and Model Path
        database_path   = os.path.join(FOLDER, f'database/{gesture_file}/Gestures/')
        self.model_path = os.path.join(FOLDER, f'model/{gesture_file}')

        # Prepare Dataloaders
        dataset_shapes = self.prepareDataloaders(database_path, cfg.batch_size, cfg.train_set_size, cfg.validation_set_size, cfg.test_set_size)

        # Create Model
        self.createModel(self.model_path, dataset_shapes, cfg.optimizer, cfg.learning_rate, cfg.loss_function)

    def prepareDataloaders(self, database_path:str, batch_size:int, train_set_size:float, validation_set_size:float, test_set_size:float) -> Tuple[torch.Size, torch.Size]:

        """ Prepare Dataloaders """

        # Load Gesture List
        gestures = np.array([gesture for gesture in os.listdir(database_path)])

        # Process Gestures
        sequences, labels = self.processGestures(database_path, gestures)    

        # Create Dataset
        dataset = GestureDataset(sequences, labels)

        # Assert Dataset Shape
        assert sequences.shape[0] == labels.shape[0], 'Sequences and Labels must have the same length'
        assert torch.Size(sequences.shape) == dataset.getInputShape(), 'Dataset Input Shape must be equal to Sequences Shape'
        assert labels.shape[0] == dataset.getOutputShape()[0], 'Dataset Output Shape must be equal to Labels Shape'    

        # Split Dataset
        assert train_set_size + validation_set_size + test_set_size <= 1, 'Train + Validation + Test Set Size must be less than 1'
        train_data, val_data, test_data = random_split(dataset, [train_set_size, validation_set_size, test_set_size], generator=torch.Generator())
        assert len(train_data) + len(val_data) + len(test_data) == len(dataset), 'Train + Validation + Test Set Size must be equal to Dataset Size'

        # Create data loaders for training and testing
        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=True)
        self.val_dataloader   = DataLoader(val_data,   batch_size=batch_size, num_workers=os.cpu_count(), shuffle=False)
        self.test_dataloader  = DataLoader(test_data,  batch_size=batch_size, num_workers=os.cpu_count(), shuffle=False)

        # Return Dataset Input and Output Shapes
        return dataset.getInputShape(), dataset.getOutputShape()

    def getDataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:

        """ Get Dataloaders """

        return self.train_dataloader, self.val_dataloader, self.test_dataloader

    def createModel(self, model_path:str, dataset_shape:Tuple[torch.Size, torch.Size], optimizer:str='Adam', lr:float=0.0005, loss_function:str='cross_entropy'):

        # Get Input and Output Sizes from Dataset Shapes
        input_size, output_size = torch.Size(list(dataset_shape[0])[1:]), torch.Size(list(dataset_shape[1])[1:])
        # print(f'\n\nInput Shape: {dataset_shape[0]} | Output Shape: {dataset_shape[1]}')
        # print(f'Input Size: {input_size} | Output Size: {output_size}')

        # Save Model Parameters
        save_parameters(model_path, 'model_parameters.yaml', input_size=list(input_size), output_size=list(output_size), optimizer=optimizer, lr=lr, loss_function=loss_function)

        # Create NeuralNetwork Model
        self.model = NeuralClassifier(input_size, output_size, optimizer, lr, loss_function)
        self.model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def getModel(self) -> Tuple[LightningModule, str]:

        """ Get NN Model and Model Path """

        return self.model, self.model_path

    def processGestures(self, database_path:str, gestures:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """ Process Gestures Dataset """

        """ Dataset:

        1 Pickle File (.pkl) for each Gesture
        Each Pickle File contains a Number of Videos Representing the Gesture
        Each Video is Represented by a Sequence of 3D Keypoints (x,y,z) for each Frame of the Video

        Dataset Structure:

            - Array of Sequences (Videos): (Number of Sequences / Videos, Sequence Length, Number of Keypoints (Flattened Array of 3D Coordinates x,y,z,v))
            - Size: (N Video, N Frames, N Keypoints) -> (1000+, 85, 300) or (1000+, 85, 1734)

            Frames: 85 (Fixed) | Keypoints (300 or 1734):

            Right Hand: 21 * 4 = 84
            Left  Hand: 21 * 4 = 84
            Pose:       33 * 4 = 132
            Face:       478 * 3 = 1434

        """

        # Loop Over Gestures
        for index, gesture in enumerate(sorted(gestures)):

            # Load File
            with open(os.path.join(database_path, f'{gesture}'), 'rb') as f:

                # Load the Keypoint Sequence (Remove First Dimension)
                try: sequence = np.array(pickle.load(f)).squeeze(0)
                except Exception as error: print(f'ERROR Loading "{gesture}": {error}'); exit(0)

                # Get the Gesture Name (Remove ".pkl" Extension)
                gesture_name = os.path.splitext(gesture)[0]

                # Get Label Array (One Label for Each Video)
                labels = np.array([index for _ in sequence]) if 'labels' not in locals() else np.concatenate((labels, np.array([index for _ in sequence])), axis=0)

                # Concatenate the Zero-Padded Sequences into a Single Array
                gesture_sequences = sequence if 'gesture_sequences' not in locals() else np.concatenate((gesture_sequences, sequence), axis=0)

                # Debug Print | Shape: (Number of Sequences / Videos, Sequence Length, Number of Keypoints (Flattened Array of 3D Coordinates x,y,z,v))
                print(f'Processing: "{gesture}"'.ljust(30), f'| Sequence Shape: {sequence.shape}'.ljust(30), f'| Label: "{gesture_name}"')

        print(colored(f'\n\nTotal Sequences Shape: ', 'yellow'), f'{gesture_sequences.shape} | ', colored('Total Labels Shape: ', 'yellow'), f'{labels.shape}\n\n')
        return gesture_sequences, labels

if __name__ == '__main__':

    main()
