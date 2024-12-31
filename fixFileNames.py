import os

import os


def rename_files(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    files.sort(key=lambda x: int(x.split('.')[0]))
    for index, filename in enumerate(files):
        # Get the file extension
        file_extension = os.path.splitext(filename)[1]
        # Create the new filename
        new_filename = f"{index}{file_extension}"
        # Get the full path for the original and new filenames
        original_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)
        # Rename the file
        os.rename(original_filepath, new_filepath)

    print(f"Renamed {len(files)} files in {folder_path}.")

# Example usage
folder_path = 'D:\TF\\reversed_saturation_test'  # Replace with your folder path
rename_files(folder_path)