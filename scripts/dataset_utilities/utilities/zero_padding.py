import os, pickle
import numpy as np
from numba import njit
from termcolor import colored

class ZeroPadding:

    def __init__(self, dataset_path:str, keypoint_number:int=300, overwrite:bool=False):

        # Dataset .pkl Files Path
        self.dataset_path = dataset_path

        # Find Max Number of Frame
        self.max_frames = self.get_max_frames()

        # Zero-Padding
        self.zeroPadding(keypoint_number, overwrite)

    def get_max_frames(self):

        """ Find the max video frame number to make the zero-padding """

        # Init Max Frame
        max_frames = 0

        # Loop Over the Gestures
        for gesture in sorted(file for file in os.listdir(self.dataset_path) if file.endswith(('.pkl'))):

            print(f'Processing: {gesture}')

            # Load the Gesture .pkl File
            with open(os.path.join(self.dataset_path, f'{gesture}'), 'rb') as f:

                # Get the Gesture Sequence
                sequence = pickle.load(f)

                # Update Max Frames Number
                max_frames = max([max_frames, max([len(array) for array in sequence])])

        print(f'\nMax Frames Number: {max_frames}\n')
        return max_frames

    @staticmethod
    @njit
    def post_padding(array, max_shape, keypoint_number:int=300):

        padded_sequence = np.zeros((int(max_shape), keypoint_number))
        padded_sequence[:array.shape[0], :array.shape[-1]] = array
        return padded_sequence

    @staticmethod
    @njit
    def pre_padding(array, max_shape, keypoint_number:int=300):

        padded_sequence = np.zeros((int(max_shape), keypoint_number))
        padded_sequence[max_shape - array.shape[0]:, :array.shape[-1]] = array
        return padded_sequence

    def zeroPadding(self, keypoint_number:int=300, overwrite:bool=False):

        """ Zero-Padding the Video Sequences """

        # Loop Over the Gestures
        for gesture in sorted(file for file in os.listdir(self.dataset_path) if file.endswith(('.pkl'))):

            # Load the Gesture .pkl File
            with open(os.path.join(self.dataset_path, f'{gesture}'), 'rb') as f:

                # Get the Gesture Sequence
                sequence = pickle.load(f)
                video_sequence = np.zeros((len(sequence), int(self.max_frames), keypoint_number))

                # Loop Over the Sequence
                for i in range(len(sequence)):

                    padded_sequence = self.pre_padding(sequence[i], self.max_frames, keypoint_number)
                    video_sequence[i] = padded_sequence

                    print(f'Processing: {gesture} | Shape: {video_sequence.shape}')

            # Zero-Padded Save File
            save_path = os.path.join(self.dataset_path, gesture) if overwrite else os.path.join(self.dataset_path, f'{gesture}_zero')

            # Save the Zero-Padded Sequence
            with open(save_path, 'wb') as f: pickle.dump([video_sequence], f)
            print(colored(f'\nZero-Padding Completed:', 'green'), f'{gesture}\n')
