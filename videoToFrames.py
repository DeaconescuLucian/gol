import cv2
import os


def extract_images(video_path, output_folder):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize variables
    frame_count = 0

    # Read the video frames
    while True:
        # Read the next frame
        ret, frame = video.read()

        # If the frame was not successfully read, we have reached the end of the video
        if not ret:
            break

        # Save the frame as an image
        image_path = os.path.join(output_folder, f"{frame_count}.jpg")
        cv2.imwrite(image_path, frame)

        # Increment the frame count
        frame_count += 1

    # Release the video file
    video.release()

    print(f"Successfully extracted {frame_count} frames.")


# Example usage
# for i in range(12):
#     video_path = f"clips_for_loop/{i}.mp4"
#     output_folder = f"clip{i}"
#     extract_images(video_path, output_folder)

video_path = f"vids/mini_sat_loop.mp4"
output_folder = f"reversed_saturation_test"
extract_images(video_path, output_folder)




