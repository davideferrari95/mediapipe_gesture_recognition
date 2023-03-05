import os, cv2, csv
import pandas as pd
from termcolor import colored

# Working Path (__file__: script/utils/dataset_converter.py)
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = f'{PROJECT_PATH}/dataset/20bn-jester-v1'
OUTPUT_PATH  = f'{PROJECT_PATH}/dataset/Video'
CSV_PATH     = f'{PROJECT_PATH}/dataset/Labels'

# -------------------------------------- MERGE CSV -------------------------------------- #

# Read in the CSV Files
df1 = pd.read_csv(f'{CSV_PATH}/Test.csv', sep=';', header=None)
df2 = pd.read_csv(f'{CSV_PATH}/Train.csv', sep=';', header=None)
df3 = pd.read_csv(f'{CSV_PATH}/Validation.csv', sep=';', header=None)

# Concatenate the DataFrames
merged_df = pd.concat([df1, df2, df3], axis=0)

# Save the Merged Dataframe to a New CSV File
merged_df.to_csv(f'{CSV_PATH}/Total.csv', index=False, header=None)
merged_df.sort_values(0).to_csv(f'{CSV_PATH}/Sorted_Total.csv', index=False, header=None)

# Print DataFrames
print(f'\n{colored("Test CSV:", "yellow")}\n\n', df1, f'\n\n\n{colored("Train CSV:", "yellow")}\n\n', df2, f'\n\n\n{colored("Validation CSV:", "yellow")}\n\n', df3, '\n')
print(f'\n{colored("Total CSV:", "yellow")}\n\n', merged_df, '\n\n')

# -------------------------------------- CONVERT DATASET -------------------------------------- #

# Define the Codec
codec = cv2.VideoWriter_fourcc(*'mp4v')

# Make a Dictionary with Each Label for Every Frame Subfolder
with open(f'{CSV_PATH}/Total.csv', 'r') as f: dataset = {int(number):label for number, label in csv.reader(f)}

# Iterate for Every Gesture Subfolder in Database
for subfolder_name in sorted(os.listdir(DATASET_PATH)):

    if os.path.isdir(os.path.join(DATASET_PATH, subfolder_name)):
       
        # Take the Label of Every Subfolder, using the Subfolder Number
        subfolder_label = dataset[int(subfolder_name)]

        # Choose the Corresponding Gesture Subfolder Path to Save the Video
        subfolder_path = os.path.join(OUTPUT_PATH, subfolder_label)

        # Create the Subfolder (if not exist)
        os.makedirs(subfolder_path, exist_ok=True)

        # Define Input - Output Paths
        input_path  = os.path.join(DATASET_PATH, subfolder_name)
        output_path = os.path.join(subfolder_path, subfolder_name + ".mp4")
       
        # Debug Print
        print(f'Subfolder: {int(subfolder_name):6}   |   Label: {subfolder_label}')
        # print(f'Input Path: {input_path} | Output Path: {output_path}')

        # Create VideoWriter Object
        out = cv2.VideoWriter(output_path, codec, len(os.listdir(input_path))/3, (176*2,100*2))
   
        # Iterate Through the Images in the Subfolder
        for filename in sorted(os.listdir(input_path)):

            # Choose Only the `.jpg File to Merge
            if filename.endswith('.jpg'):

                # Read, Resize and Write the Image to the VideoWriter
                img = cv2.imread(os.path.join(input_path, filename))
                img = cv2.resize(img, (176*2, 100*2))
                out.write(img)

        # Release the VideoWriter
        out.release()

print("\nAll Video Processed")
