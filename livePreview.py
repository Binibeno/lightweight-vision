import cv2
from picamera2 import Picamera2
from libcamera import Transform
import time
print("init")
print("VNC / Monitor does have to be on in order to this to work.")


# Initialize the Picamera2
picam2 = Picamera2()
# default
# picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.size = (640, 310)


picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.preview_configuration.transform=Transform(vflip=1)
picam2.configure("preview")
picam2.start()

while True:
    frame = picam2.capture_array()
  
    # Display the resulting frame
    cv2.imshow("Camera", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        print("")
        break

# Release resources and close windows
cv2.destroyAllWindows()
