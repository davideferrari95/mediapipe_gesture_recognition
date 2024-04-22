import pickle
import os
import numpy as np
from numba import njit

class DatasetConverter:

    def __init__(self):

        
        self.datasetpath = '/home/alberto/catkin_ws/src/mediapipe_gesture_recognition/database/3D_Gestures/RightLeftPose/ToPad_file'
        self.zerodatasethpath = '//home/alberto/catkin_ws/src/mediapipe_gesture_recognition/database/3D_Gestures/RightLeftPose/Padded_file'

    def findShape(self, gestures):      

        '''
        Function created to find the max video frame number to make the zero-padding 
        '''

        #Initialise the max number of frame
        new_max = 0
        
        #Loop over the gestures
        for gesture in sorted(gestures):
            
            #Get the path of the pkl file
            load_file_path = os.path.join(self.datasetpath, f'{gesture}')

            #Load the pkl file
            with open(load_file_path, 'rb') as f:
                
                #Get the gesture sequence
                sequence = pickle.load(f)
                
                #Find the max number of frame
                max_x = max([len(array) for array in sequence])
                
                #Update the max number of frame
                new_max  = max([new_max, max_x])
        
        print("Max number of frame:", new_max)

        return new_max

    @staticmethod
    @njit
    def zero_padding(array, max_shape):

        padded_sequence = np.zeros((int(max_shape), 300))
        padded_sequence[:array.shape[0], :array.shape[-1]] = array
        return padded_sequence

    def zeroPadding(self):

        gestures = [f for f in os.listdir(self.datasetpath)]

        max_shape = self.findShape(gestures)

        for gesture in sorted(gestures):

            # Get the path of the pkl file

            self.picklefilepath = os.path.join(self.datasetpath, f'{gesture}')

            # Load File
            with open(self.picklefilepath, 'rb') as f:

                # Get gesture sequence from pkl file
                sequence = pickle.load(f)
                video_sequence = np.zeros((len(sequence), int(max_shape), 300))

                # Loop over the sequence
                for i in range(len(sequence)):

                    padded_sequence = self.zero_padding(sequence[i], max_shape)
                    video_sequence[i] = padded_sequence

                    print("I'm processing |", gesture, "And the array now is", video_sequence.shape)
            
            # Save the zero-padded sequence
            

            zeropicklefilepath = os.path.join(self.zerodatasethpath, gesture)
 
            with open(zeropicklefilepath, 'wb') as f:
                pickle.dump([video_sequence], f)

            print(gesture, "Done")

if __name__ == '__main__':

    
    DC = DatasetConverter()

    print("\nSTART CONVERTING PHASE\n")
    # Train Network
    DC.zeroPadding()
    print("\n END CONVERTING PHASE\n")
