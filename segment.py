from picamera2 import Picamera2
from libcamera import Transform



from picamera2 import Picamera2

from ultralytics import YOLO
from libcamera import Transform
from depthest import init, estimate
pi_image = None

print("importing done")

picam2 = Picamera2()
# picam2.preview_configuration.main.size = (1280, 720)
# picam2.preview_configuration.main.size = (640, 310)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.preview_configuration.transform=Transform(vflip=1)

picam2.start()  

# Capture the image and save it to the 'array' variable
frame = picam2.capture_array("main") 

# Stop the camera to free up resources
picam2.stop() 
picam2.stop_encoder()

del picam2

segmentModel = YOLO("yolo11n-seg.pt")  

results = segmentModel(frame)


# Process results lists
for result in results:
  print("Running")
  masks = result.masks  # Masks object for segmentation masks outputs
  print(masks)
  # result.show()  # display to screen
  result.save(filename="output/segmented.jpg")  # save to disk