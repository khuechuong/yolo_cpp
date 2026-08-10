# YOLO C++
A lightweight, high-speed C++ deployment framework designed to run [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO instance segmentation models using the ONNX Runtime C++ API with native CUDA acceleration.

## Installation

Check what cuda you have 12 or 13 with ```nvidia-smi```

```
~$ nvidia-smi
Tue Jul 21 13:23:46 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.05              Driver Version: 595.71.05      CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2070 ...    Off |   00000000:01:00.0  On |                  N/A |
| 39%   36C    P8             12W /  215W |     931MiB /   8192MiB |      3%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1689      G   /usr/lib/xorg/Xorg                      401MiB |
|    0   N/A  N/A            1852      G   /usr/bin/gnome-shell                     53MiB |
|    0   N/A  N/A            3260      G   .../8595/usr/lib/firefox/firefox        329MiB |
|    0   N/A  N/A          594932      G   ...rack-uuid=3190708998493415531        126MiB |
+-----------------------------------------------------------------------------------------+

```

Make sure cudatookit and cudnn of same version is included as well. For example, my cuda is 13.2 with ubuntu 22.04
```
# example wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/<distro>/<arch>/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
# toolkit & cudnn
sudo apt install cuda-toolkit-13-2
sudo apt install libcudnn9-cuda-13
# if everything is in x86_64-linux-gnu
sudo ldconfig /usr/lib/x86_64-linux-gnu
sudo ldconfig
```

Check which version you want on this [website](https://github.com/microsoft/onnxruntime/releases) and install tgz file. For example, I'm using ```v1.27.1``` with ```cuda 13```:
```
wget https://github.com/microsoft/onnxruntime/releases/download/v1.27.1/onnxruntime-linux-x64-gpu_cuda13-1.27.1.tgz

tar -xzf onnxruntime-linux-x64-gpu_cuda13-1.27.1.tgz

sudo cp -r onnxruntime-linux-x64-gpu_cuda13-1.27.1/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-gpu_cuda13-1.27.1/lib/* /usr/local/lib/
sudo ldconfig
```
## Build
```
git clone https://github.com/khuechuong/yolo_cpp.git
cd yolo_cpp
mkdir build && cd build
cmake ..
make -j4
```

## Ultralytics model conversion
Convert your <>.pt file you got from training with [Ultralytics](https://github.com/ultralytics/ultralytics) with
```python
from ultralytics import YOLO

model = YOLO(<MODEL_PATH>)  # your trained weights
model.export(format='onnx', imgsz=<YOUR MODEL PARAM>, opset=<YOUR MODEL PARAM>)
```
* NOTE: the ```.export``` has more params that you need to do reading to make sure your output and class is correct so read more in [Ultralytics](https://github.com/ultralytics/ultralytics) API before you do any conversion.

#### Debug & Verify
Check if ldconfig know where libcudnn is
```
ldconfig -p | grep libcudnn
```

if it does, find where it is
```
find /usr -name "libcudnn*.so*" 2>/dev/null
find /lib -name "libcudnn*.so*" 2>/dev/null
```

if there are in ```<>```, then ```sudo ldconfig <>```
example:
```
sudo ldconfig /usr/lib/x86_64-linux-gnu
sudo ldconfig
```
verify:
```
ldconfig -p | grep libcudnn
```
## CMakeList

Add these 2 lines into your CMakeList.txt
```
include_directories(/usr/local/include/onnxruntime)
link_directories(/usr/local/lib)
```

and have ```onnxruntime``` inside your ```target_link_libraries```

```
target_link_libraries(<node>
    onnxruntime
    <other dep>
    <other dep>
)
```

## C++

Include:

```c++
#include <onnxruntime_cxx_api.h>
```

Config
```c++
const int   INPUT_W     = 640;
const int   INPUT_H     = 640;
const float CONF_THRESH = 0.25f;
const int   MASK_W      = 160;
const int   MASK_H      = 160;
const int   MASK_COEFF  = 32;

const std::vector<std::string> CLASS_NAMES = {
    "spall", "rebar", "algae", "fiducial_marker", "crack", "eff"
};

const std::vector<cv::Scalar> CLASS_COLORS = {
    {0,   140, 255},
    {0,   0,   220},
    {50,  205, 50 },
    {238, 130, 238},
    {30,  30,  220},
    {0,   215, 255},
};
```
Struct & Helpers
```c++
// ── Structs ───────────────────────────────────────────────────────────────────
struct Detection {
    cv::Rect           box;
    float              confidence;
    int                classId;
    std::vector<float> maskCoeffs;
};

// ── Helpers ───────────────────────────────────────────────────────────────────
std::vector<float> preprocess(const cv::Mat& img, float& scale,
                               int& padW, int& padH)
{
    int origW = img.cols, origH = img.rows;
    scale = std::min((float)INPUT_W / origW, (float)INPUT_H / origH);
    int newW = (int)(origW * scale);
    int newH = (int)(origH * scale);
    padW = (INPUT_W - newW) / 2;
    padH = (INPUT_H - newH) / 2;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(newW, newH));

    cv::Mat padded(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(padW, padH, newW, newH)));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    std::vector<float> data;
    data.reserve(3 * INPUT_H * INPUT_W);
    for (auto& ch : channels)
        data.insert(data.end(), (float*)ch.datastart, (float*)ch.dataend);
    return data;
}
```

CPU
```c++
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_seg");
Ort::SessionOptions sessionOpts;
sessionOpts.SetIntraOpNumThreads(4);
sessionOpts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
```
Add if GPU

```c++
OrtCUDAProviderOptions cudaOpts{};
cudaOpts.device_id = 0;
sessionOpts.AppendExecutionProvider_CUDA(cudaOpts);
```
Start session
```c++
Ort::Session session(env, modelPath.c_str(), sessionOpts);
```
Run YOLO
```c++
    // ── Build input tensor ─────────────────────────────────────────────────
    std::vector<int64_t> inputShape = {1, 3, INPUT_H, INPUT_W};
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        inputData.data(), inputData.size(),
        inputShape.data(), inputShape.size()
    );

    // ── Inference ──────────────────────────────────────────────────────────
    const char* inputNames[]  = {"images"};
    const char* outputNames[] = {"output0", "output1"};

    // Warm-up: CUDA/TRT compiles kernels on the first call — discard that cost
    std::cout << "Warm-up inference...\n";
    {
        auto warmup = session.Run(
            Ort::RunOptions{nullptr},
            inputNames, &inputTensor, 1,
            outputNames, 2
        );
    }
    std::cout << "Warm-up done. Timed run...\n";
    
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        inputNames, &inputTensor, 1,
        outputNames, 2
    );

    float* out0Data = outputs[0].GetTensorMutableData<float>();
    float* out1Data = outputs[1].GetTensorMutableData<float>();
```

Full code at [main.cpp](main.cpp)

# Run
```
cd yolo_cpp
./build/yolo_culvert cuda <MODEL_PATH> <IMAGE_PATH> <OUTPUT_PATH>

```
