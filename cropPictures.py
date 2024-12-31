import cv2

gen = 0
while gen < 574:
    # Load the image
    input_image_path = f'infinite_loop_edge/{gen}.jpg'  # Replace with the path to your image file
    output_image_path = f'infinite_loop_edge/{gen}.jpg'  # Path to save the cropped image
    image = cv2.imread(input_image_path)

    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image at {input_image_path}")
    else:
        # Get the dimensions of the image
        height, width, channels = image.shape

        # Define the cropping coordinates
        x_start = 420
        x_end = 1500

        # Check if the cropping coordinates are within the image dimensions
        if x_end <= width and x_start >= 0 and x_start < x_end:
            # Crop the image
            cropped_image = image[:, x_start:x_end]

            # Save the cropped image
            cv2.imwrite(output_image_path, cropped_image)

            # Optionally, display the original and cropped images
            cv2.imshow('Cropped Image', cropped_image)

        else:
            print("Error: Cropping coordinates are out of bounds.")
    gen += 1
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
