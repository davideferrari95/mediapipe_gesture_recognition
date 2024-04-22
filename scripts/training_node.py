#!/usr/bin/env python3

import sys, os, signal, logging
import pickle, numpy as np
from typing import Union, Tuple, Optional, List
from termcolor import colored

# Import PyTorch Lightning
import torch, pytorch_lightning as pl
from torchmetrics.classification import Accuracy
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.callbacks import EarlyStopping

# Import PyTorch Lightning Utilities
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
    trainer = pl.Trainer(

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

        print(colored('One-Hot Encoding:\n\n', 'yellow'), f'{self.y}\n')

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

class NeuralClassifier(pl.LightningModule):

    """ Classifier Neural Network """

    def __init__(self, input_shape, output_shape, optimizer='AdamW', lr=0.0005, loss_function='cross_entropy'):

        super(NeuralClassifier, self).__init__()

        # Compute Input and Output Sizes
        self.input_size, self.num_classes = input_shape[1], output_shape[0]
        self.hidden_size, self.num_layers = 512, 3

        print(colored('Input Shape: ', 'yellow'), f'{input_shape} | ', colored('Output Shape: ', 'yellow'), output_shape)
        print(colored('Input Size: ', 'yellow'), f'{self.input_size} | ', colored('Num Classes: ', 'yellow'), f'{self.num_classes} | ', colored('Hidden Size: ', 'yellow'), f'{self.hidden_size} | ', colored('Num Layers: ', 'yellow'), self.num_layers)
        print(colored('Optimizer: ', 'yellow'), f'{optimizer} | ', colored('Learning Rate: ', 'yellow'), f'{lr} | ', colored('Loss Function: ', 'yellow'), f'{loss_function}\n\n')

        # Create LSTM Layers (Input Shape = Number of Flattened Keypoints (300 / 1734))
        self.lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=False).to(DEVICE)

        # Create Fully Connected Layers
        self.net = self.mlp(self.hidden_size, self.num_classes, hidden_size=[256,128], hidden_mod=torch.nn.ReLU()).to(DEVICE)

        # Initialize Accuracy Metrics
        self.train_accuracy, self.test_accuracy, self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes), Accuracy(task="multiclass", num_classes=self.num_classes), Accuracy(task="multiclass", num_classes=self.num_classes)

        # Instantiate Loss Function and Optimizer
        self.loss_function = getattr(torch.nn.functional, loss_function)
        self.optimizer     = getattr(torch.optim, optimizer)
        self.learning_rate = lr

        # Initialize Validation and Test Losses
        self.val_loss,   self.num_val_batches  = 0, 0
        self.test_loss,  self.num_test_batches = 0, 0

        print(colored(f'Model Initialized:\n\n', 'green'), self, '\n')

        # exit()

    # Neural Network Creation Function
    def mlp(self, input_size:int, output_size:int, hidden_size:Optional[List[int]]=[512,256], hidden_mod:Optional[torch.nn.Module]=torch.nn.ReLU(), output_mod:Optional[torch.nn.Module]=None):

        ''' Neural Network Creation Function '''

        # No Hidden Layers
        if hidden_size is None or hidden_size == []:

            # Only one Linear Layer
            net = [torch.nn.Linear(input_size, output_size)]

        else:

            # First Layer with ReLU Activation
            net = [torch.nn.Linear(input_size, hidden_size[0]), hidden_mod]

            # Add the Hidden Layers
            for i in range(len(hidden_size) - 1):
                net += [torch.nn.Linear(hidden_size[i], hidden_size[i+1]), hidden_mod]

            # Add the Output Layer
            net.append(torch.nn.Linear(hidden_size[-1], output_size))

        if output_mod is not None:
            net.append(output_mod)

        # Create a Sequential Neural Network
        return torch.nn.Sequential(*net)

    def forward(self, x:torch.Tensor):

        """ Forward Pass """

        # Pass through LSTM Layer
        lstm_out, _ = self.lstm(x)
        lstm_out = torch.nn.functional.relu(lstm_out)
        lstm_out = torch.nn.functional.dropout(lstm_out, p=0.5)

        # Only Last Time Step Output
        return self.net(lstm_out[:, -1, :])

    def compute_loss(self, batch:Tuple[torch.Tensor, torch.Tensor], log_name:str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """ Compute Loss """

        # Get X,Y from Batch
        x, y = batch

        # Forward Pass
        y_pred:torch.Tensor = self(x)

        # Compute Loss
        loss:torch.Tensor = self.loss_function(y_pred, y.float())
        self.log(log_name, loss)

        return loss, y_pred, y

    def training_step(self, batch:Tuple[torch.Tensor, torch.Tensor], batch_idx):

        """ Training Step """

        # Compute Loss
        loss, y_pred, y = self.compute_loss(batch, 'train_loss')

        # Update Accuracy Metric
        self.train_accuracy(y_pred, y)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch:Tuple[torch.Tensor, torch.Tensor], batch_idx):

        """ Validation Step """

        # Compute Loss
        val_loss, y_pred, y = self.compute_loss(batch, 'val_loss')

        # Update Accuracy Metric
        self.val_accuracy(y_pred, y)

        # Update Validation Loss
        self.val_loss += val_loss.item()
        self.num_val_batches += 1
        return val_loss

    def on_validation_epoch_end(self):

        """ Validation Epoch End """

        # Calculate Average Validation Loss
        avg_val_loss = self.val_loss / self.num_val_batches
        self.log('val_loss', avg_val_loss)
        self.log('val_accuracy', self.val_accuracy.compute(), prog_bar=True)

    def test_step(self, batch:Tuple[torch.Tensor, torch.Tensor], batch_idx):

        """ Test Step """

        # Compute Loss
        test_loss, y_pred, y = self.compute_loss(batch, 'test_loss')

        # Update Accuracy Metric
        self.test_accuracy(y_pred, y)

        # Update Test Loss
        self.test_loss += test_loss.item()
        self.num_test_batches += 1
        return test_loss

    def on_test_epoch_end(self):

        """ Test Epoch End """

        # Calculate Average Test Loss
        avg_test_loss = self.test_loss / self.num_test_batches
        self.log('test_loss', avg_test_loss)
        self.log('test_accuracy', self.test_accuracy.compute(), prog_bar=True)

    def configure_optimizers(self):

        """ Configure Optimizer """

        # Return Optimizer
        return self.optimizer(self.parameters(), lr = self.learning_rate)

class GestureRecognitionTraining3D:

    """ 3D Gesture Recognition Training Class """

    def __init__(self, cfg:Params):

        # Choose Gesture File
        gesture_file = ''
        if cfg.enable_right_hand: gesture_file += 'Right'
        if cfg.enable_left_hand:  gesture_file += 'Left'
        if cfg.enable_pose:       gesture_file += 'Pose'
        if cfg.enable_face:       gesture_file += 'Face'
        print(colored(f'\n\nLoading: {gesture_file} Configuration\n', 'yellow'))

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

        # Save Model Parameters
        save_parameters(model_path, 'model_parameters.yaml', input_size=list(input_size), output_size=list(output_size), optimizer=optimizer, lr=lr, loss_function=loss_function)

        # Create NeuralNetwork Model
        self.model = NeuralClassifier(input_size, output_size, optimizer, lr, loss_function)
        self.model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def getModel(self) -> Tuple[pl.LightningModule, str]:

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
                print(f'Processing: "{gesture}"'.ljust(30), f'| Sequence Shape: {sequence.shape}'.ljust(30), f'| Label: {labels[-1]} | Gesture: "{gesture_name}"')

        print(colored(f'\nTotal Sequences Shape: ', 'yellow'), f'{gesture_sequences.shape} | ', colored('Total Labels Shape: ', 'yellow'), f'{labels.shape}\n\n')
        return gesture_sequences, labels

if __name__ == '__main__':

    main()
