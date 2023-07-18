import cv2
from ultralytics import YOLO


model = YOLO("models/yolov8n.pt")
image = cv2.imread("images/bus.jpg")
results = model.predict(source=image, save=True, save_txt=False)
# results = model.predict(source=0, show=True)

