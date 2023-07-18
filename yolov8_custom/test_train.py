from ultralytics import YOLO

MODEL = 'models/yolov8n.pt'
CFG = 'yolov8n.yaml'
HOME = '/mnt/data/workspace/yolov8/'

model = YOLO(MODEL)                   # Load pre-trained model
# model = YOLO(CFG)                       # Train from scratch

# # Freeze backbone
# # ================================================================== #
n_freeze = 10
freeze = [f'model.{x}.' for x in range(n_freeze)]  # layers to freeze 
for k, v in model.model.named_parameters(): 
    v.requires_grad = True  # train all layers 
    if any(x in k for x in freeze): 
        print(f'freezing {k}') 
        v.requires_grad = False
# # ================================================================== #

model.train(data=HOME+'data.yaml', patience=1000, imgsz=320)