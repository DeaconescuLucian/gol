import cv2
import numpy as np

def detect_objects(image_path):
    # Load the pre-trained model
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

    # Load the input image
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # Preprocess the image for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # Set the input to the neural network
    net.setInput(blob)

    # Run forward pass and get the output
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and label on the image
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = f"{confidence * 100:.2f}%"
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

# Call the function to detect objects in an image
detect_objects('goldfire.png')