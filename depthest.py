from PIL import Image
from transformers import pipeline
import numpy as np

pipe = None

def init():
    global pipe
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    print("depth est. init")

def estimate(picam_array, bottleBox):
    pi_image = Image.fromarray(picam_array)
    # inference
    depth = pipe(pi_image)["depth"]
    depth.save("output/depth_estimation.png")

    # calculate the depth of the bottle
    # calculate average depth of pixels inside bounding box
    x1, y1, x2, y2 = bottleBox.xyxy.cpu().numpy()[0]
    bottle_depth = np.array(depth)[int(y1):int(y2), int(x1):int(x2)]
    average_depth = np.mean(bottle_depth)
    print(f"Average depth of the bottle: {average_depth}")

    return average_depth

