import cv2
from ultralytics import YOLO

# train_type = "freeze"
# train_type = "scratch"
train_type = "pretrain"
folder = "3d_shapes/test/images"

models_v8_1000 = f"v8_1000_{train_type}"
model = YOLO(f"models_v8_1000/{models_v8_1000}.pt")

results = model.predict(source=folder, save=True, save_txt=True, save_conf=True)
