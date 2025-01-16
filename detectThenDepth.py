import cv2
from picamera2 import Picamera2

from ultralytics import YOLO
from libcamera import Transform
import time
import numpy as np
from depthest import init, estimate
print("init")

# Initialize the Picamera2
picam2 = Picamera2()
# default
# picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.size = (640, 310)
# picam2.preview_configuration.main.size = (640 /2, 310/2)

picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.preview_configuration.transform=Transform(vflip=1)
picam2.configure("preview")
picam2.start()


# Load the YOLO11 model
# have to run create model first 
model = YOLO("yolo11n_ncnn_model", task="detect", )

instantBreak = False


# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0
# font which we will be using to display FPS 
font = cv2.FONT_HERSHEY_SIMPLEX 

bottleFrameDetected = None
bottleFrameNew = None

results = None
bottleBox = None

while True:
    # time when we finish processing for this frame 
    # ! do all processing below this, and above the fps calculator
    new_frame_time = time.time() 

    # Capture frame-by-frame
    frame = picam2.capture_array()

    # Run YOLO11 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    # annotated_frame = results[0].plot()

    new_frame_time = time.time() 
  
    # Calculating the fps 
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    fps = int(fps) 
    fps = str(fps) 
    # cv2.putText(annotated_frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
    print(fps)
    # Check if a bottle has been recognized
    for result in results:
        # probs = result.probs  # Probs object for classification outputs
        # print(probs)
        # if probs != None:
            # print("Probs", probs)
            # break;
        for detection in result.boxes:
        #     # Assuming detection.cls is an integer index for the class
            bottleBox = detection
            if detection.cls == 39:  # correct class index for "bottle"
                print("Bottle recognized", )
                print("Probabilty", detection.conf)
                bottleFrameDetected = frame
                instantBreak = True
                break


    # Display the resulting frame
    # cv2.imshow("Camera", annotated_frame)

    if instantBreak:
        break
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        print("")
        break

# Release resources and close windows
cv2.destroyAllWindows()

# https://docs.ultralytics.com/modes/predict/#__tabbed_1_1
if bottleFrameDetected is not None:
    # Capture a new frame of the bottle
    # should not be blurry
    bottleFrameNew = picam2.capture_array()
    cv2.imwrite("output/bottle_new.png", bottleFrameNew)

# Process results list
# results for bottleFrameDetected
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="output/bottle_detected.png")  # save to disk

init()
estimate(bottleFrameDetected, bottleBox)
