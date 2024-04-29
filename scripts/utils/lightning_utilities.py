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
