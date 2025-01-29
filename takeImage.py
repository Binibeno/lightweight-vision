from picamera2 import Picamera2
from libcamera import Transform
from matplotlib import pyplot as plt
from PIL import Image

pi_image = None

# Initialize the Picamera2
with Picamera2() as picam2:
  # picam2.preview_configuration.main.size = (1280, 720)
  # picam2.preview_configuration.main.size = (640, 310)
  picam2.preview_configuration.main.size = (640, 310)
  picam2.preview_configuration.main.format = "BGR888"
  picam2.preview_configuration.align()
  picam2.preview_configuration.transform=Transform(vflip=1)
  picam2.start()  
  # Capture the image and save it to the 'array' variable
  array = picam2.capture_array("main") 

  # Stop the camera to free up resources
  picam2.stop() 
  picam2.stop_encoder()

  del picam2


  # # Display the captured image
  # plt.imshow(array)
  # plt.axis('off')  # Hide the axis
  # plt.show()
  pi_image = Image.fromarray(array)
  pi_image.save('./output/temp_image.png')

