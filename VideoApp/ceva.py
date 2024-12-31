import cv2
import pyautogui

# Open the cv2 window
cv2.namedWindow('Window')

while True:
    # Capture a frame
    ret, frame = capture.read()

    # Display the frame in the cv2 window
    cv2.imshow('Window', frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF

    # If 'q' is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # If other keys are pressed, send them to the cv2 window
    if key != 255:
        pyautogui.press(chr(key))

# Release the capture and destroy the cv2 window
capture.release()
cv2.destroyAllWindows()