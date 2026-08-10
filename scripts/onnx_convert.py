from ultralytics import YOLO

model = YOLO('/home/chuong/Documents/Spall_Volume/model/yolov26l-seg-full-s2ds-e.v3.pt')  # your trained weights
model.export(format='onnx', imgsz=640, opset=12)