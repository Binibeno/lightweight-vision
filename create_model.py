from ultralytics import YOLO

# for detection: 

# Load a YOLO11n PyTorch model
model = YOLO("yolo11n.pt")

# Export the model to NCNN format
# the NCNN format is the fastest on the Raspberry Pi
model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'


# for classification
# https://docs.ultralytics.com/tasks/classify/#models
model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

# performed bad. and did not even work.
model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'
print("Dont use ncnn classification model. performed badly during testing.")