import os
import cv2, csv

# Gesture frames folder path
root_path = '/home/alberto/catkin_ws/src/mediapipe_gesture_recognition/dataset/Gesture_frames'

# Labeled video folder path
video_with_labels_path = '/home/alberto/catkin_ws/src/mediapipe_gesture_recognition/dataset/Video_with_labels'

#Dataset with Labels
data_file = "/home/alberto/catkin_ws/src/mediapipe_gesture_recognition/dataset/Labels/Total.csv"

# Define the codec (non so cosa sia)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Load Dataset
dataset = {}

#Make a dictionary with each label for every frame subfolder
with open(data_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        number, label = row
        dataset[int(number)] = label

# Iterate for every gesture frame subfolder
for subfolder_name in os.listdir(root_path):
    if os.path.isdir(os.path.join(root_path, subfolder_name)):
       
       # Take the subfolder number (every subfolder had a "number name")
       subfolder_number = int(subfolder_name)
          
       # Take the label of every subfolder 
       subfolder_label = dataset[subfolder_number]
   
       # Choose the right subfolder path to save the videos
       subfolder_path = os.path.join(video_with_labels_path, subfolder_label)

       # Create the subfolder if it doesn't exist
       os.makedirs(subfolder_path, exist_ok=True)

       # Define the input and output paths
       input_path = os.path.join(root_path, subfolder_name)

       output_path = os.path.join(subfolder_path, subfolder_name + ".avi")
       
       #Debug print
       print("\nI'm taking the frames from the subfolder:", subfolder_number, "in this path", input_path, "\n")
       print("I'm make the video in the subfolder", subfolder_label, "in this path", output_path, "\n")
       

       # Create VideoWriter object
       out = cv2.VideoWriter(output_path, fourcc, len(os.listdir(input_path))/3, (176*2,100*2))
   
       # Iterate through the images in the subfolder
       for filename in sorted(os.listdir(input_path)):

           #Choose only the .jpg file to merge 
           if filename.endswith('.jpg'):

               # Read the image
               img = cv2.imread(os.path.join(input_path, filename))
               # Resize the image
               img = cv2.resize(img, (176*2, 100*2))
               # Write the image to the VideoWriter
               out.write(img)



       # Release the VideoWriter
       out.release()
#print the end 
print("All done\n")
