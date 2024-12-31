import cv2
import os
import re

# Directory containing the input images
# input_images_directory = 'devil/'

def extract_number(filename):
    # Extract the number from the filename using regular expressions
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return 0

def sort_filenames(filenames):
    # Sort the filenames based on the extracted number
    sorted_filenames = sorted(filenames, key=extract_number)
    return sorted_filenames

def is_png_image(image_path):
    file_extension = os.path.splitext(image_path)[1].lower()
    return file_extension == ".png"


def delete_directory_contents(directory):
    # Iterate over all files and subdirectories in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if the current item is a file
        if os.path.isfile(file_path):
            # Delete the file
            os.remove(file_path)
        # Check if the current item is a subdirectory
        elif os.path.isdir(file_path):
            # Recursively delete the subdirectory and its contents
            delete_directory_contents(file_path)
            # Delete the empty subdirectory
            os.rmdir(file_path)
    # os.rmdir(directory)


def create(input_images_directory, name, hd=False):
    # Get the list of image filenames in the directory
    print(os.listdir(input_images_directory))
    image_files = sorted(os.listdir(input_images_directory))
    image_files = sort_filenames(image_files)
    # Determine the size of the first image (assumed to be the same for all images)
    next=5
    i=0
    if len(image_files):
        image_path = os.path.join(input_images_directory, image_files[0])
        image = cv2.imread(image_path)
        # if hd == True:
        #     image = cv2.resize(image, (1080, 1080), interpolation=cv2.INTER_AREA)
        height, width, _ = image.shape
        # Define the output video path and properties
        output_video_path = 'vids/{name}.mp4'.format(name=name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 60
        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Iterate over the image files and write each frame to the video
        # for j in range(10):
        #     print("loop" + str(j+1))
        for image_file in image_files:
                # if i == next:
                    print(image_file)
                    image_path = os.path.join(input_images_directory, image_file)
                    image = cv2.imread(image_path)
                    output_video.write(image)
                    # next += 6
                # i+=1


        # Release the video writer and close the video file
        output_video.release()
        print("Cleaning up.")
        #delete_directory_contents(input_images_directory)
    else:
        print("No images found.")


# for i in range(11):
#     create(f"clip{i}", f"clip{i}")
create(f"reversed_saturation_test", f"saturation260", False)
# create(f"edited-clip-collage-cell128-multi", f"135-190-255-05-collage-zoom")