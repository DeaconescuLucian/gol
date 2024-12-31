import os
import shutil

# Define the directory containing the images
directory = 'D:\\TF\\saturation_test'
new_directory = 'D:\\TF\\reversed_saturation_test'

# Get the list of all files in the directory
files = [f for f in os.listdir(directory) if f.endswith('.jpg')]

# Sort files to ensure they are in order
files.sort(key=lambda x: int(x.split('.')[0]))
print(files)
# Calculate the total number of files
total_files = len(files)
current_number = total_files - 1
# Rename the files by reversing their number
for file in files:
    new_filename = f'{current_number}.jpg'
    # Construct the full old and new file paths
    old_file_path = os.path.join(directory, file)
    new_file_path = os.path.join(new_directory, new_filename)
    # Rename the file
    shutil.copy(old_file_path, new_file_path)
    current_number = current_number - 1

print("Renaming completed.")