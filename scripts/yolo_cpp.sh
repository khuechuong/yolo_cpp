PROJECT_DIR="$HOME/Documents/yolo_cpp"
# cuda (works), fp16 (TODO), int8 (TODO)
MODE=cuda 

MODEL_PATH="$PROJECT_DIR/model/yolov26l-seg-full-s2ds-e.v3.onnx"
IMAGE_PATH="$PROJECT_DIR/images/000001_rgb.png"

$PROJECT_DIR/build/yolo_culvert "$MODE" "$MODEL_PATH" "$IMAGE_PATH" "output.jpg"