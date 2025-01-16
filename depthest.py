from PIL import Image
from transformers import pipeline
import numpy as np

pipe = None

def init():
    global pipe
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    print("depth est. init")

def estimate(picam_array, bottleBox, masks):
    pi_image = Image.fromarray(picam_array)
    # inference
    depth = pipe(pi_image)["depth"]
    depth.save("output/depth_estimation.png")

    # calculate the depth of the bottle
    # calculate average depth of pixels inside mask
    depth_array = np.array(depth)
    mask_array = masks.data.cpu().numpy().squeeze().astype(bool)
    bottle_depth = depth_array[mask_array]
    average_depth = np.mean(bottle_depth)
    print(f"Average depth of the bottle: {average_depth}")

    return average_depth

