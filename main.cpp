#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>       // ← timing
#include <filesystem>   // ← calibration file check (C++17)

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// ── Timing helper ─────────────────────────────────────────────────────────────
using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

inline TimePoint now() { return Clock::now(); }

inline double ms(TimePoint start, TimePoint end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct TimingReport {
    std::string mode    = "cuda";   // ← printed in header row
    double imageLoad    = 0;
    double preprocess   = 0;
    double sessionInit  = 0;
    double inference    = 0;
    double postprocess  = 0;
    double maskDraw     = 0;
    double boxDraw      = 0;
    double fileSave     = 0;
    double total        = 0;

    void print() const {
        const int W = 28;
        auto bar = [](double val, double tot, int width = 20) {
            int filled = (tot > 0) ? (int)(val / tot * width) : 0;
            return std::string(filled, '|') + std::string(width - filled, '.');
        };
        std::cout << "\n╔══════════════════════════════════════════════╗\n";
        printf(      "║  Mode: %-36s  ║\n", mode.c_str());
        std::cout <<   "╠══════════════════════════════════════════════╣\n";
        printf("║  %-*s %8.2f ms  [%s] ║\n", W, "Image load",    imageLoad,    bar(imageLoad,    total).c_str());
        printf("║  %-*s %8.2f ms  [%s] ║\n", W, "Preprocess",    preprocess,   bar(preprocess,   total).c_str());
        printf("║  %-*s %8.2f ms  [%s] ║\n", W, "Session init",  sessionInit,  bar(sessionInit,  total).c_str());
        printf("║  %-*s %8.2f ms  [%s] ║\n", W, "Inference",     inference,    bar(inference,    total).c_str());
        printf("║  %-*s %8.2f ms  [%s] ║\n", W, "Postprocess",   postprocess,  bar(postprocess,  total).c_str());
        printf("║  %-*s %8.2f ms  [%s] ║\n", W, "Mask draw",     maskDraw,     bar(maskDraw,     total).c_str());
        printf("║  %-*s %8.2f ms  [%s] ║\n", W, "Box draw",      boxDraw,      bar(boxDraw,      total).c_str());
        printf("║  %-*s %8.2f ms  [%s] ║\n", W, "File save",     fileSave,     bar(fileSave,     total).c_str());
        std::cout << "╠══════════════════════════════════════════════╣\n";
        printf("║  %-*s %8.2f ms                     ║\n", W, "TOTAL", total);
        printf("║  %-*s %8.2f FPS (inference only)   ║\n", W, "Throughput", 1000.0 / inference);
        printf("║  %-*s %8.2f FPS (end-to-end)       ║\n", W, "Throughput", 1000.0 / total);
        std::cout << "╚══════════════════════════════════════════════╝\n\n";
    }
};

// ── Config ────────────────────────────────────────────────────────────────────
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

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

std::vector<Detection> postprocess(float* data,
                                    float scale, int padW, int padH,
                                    int origW, int origH)
{
    std::vector<Detection> detections;
    const int NUM_DETS = 300;
    const int ROW_SIZE = 38;

    for (int i = 0; i < NUM_DETS; i++) {
        float* row = data + i * ROW_SIZE;

        float conf  = row[4];
        if (conf < CONF_THRESH) continue;

        int classId = (int)row[5];

        float x1 = std::max(0.0f, (row[0] - padW) / scale);
        float y1 = std::max(0.0f, (row[1] - padH) / scale);
        float x2 = std::min((float)origW, (row[2] - padW) / scale);
        float y2 = std::min((float)origH, (row[3] - padH) / scale);

        if (x2 <= x1 || y2 <= y1) continue;

        Detection det;
        det.box        = cv::Rect(cv::Point((int)x1, (int)y1),
                                  cv::Point((int)x2, (int)y2));
        det.confidence = conf;
        det.classId    = classId;
        det.maskCoeffs = std::vector<float>(row + 6, row + 6 + MASK_COEFF);

        detections.push_back(det);
    }
    return detections;
}

void applyMask(cv::Mat& result, const Detection& det,
               float* protoData,
               float scale, int padW, int padH,
               int origW, int origH)
{
    cv::Scalar color = CLASS_COLORS[det.classId % CLASS_COLORS.size()];

    cv::Mat mask(MASK_H, MASK_W, CV_32F, cv::Scalar(0));
    for (int k = 0; k < MASK_COEFF; k++) {
        float coeff = det.maskCoeffs[k];
        float* proto = protoData + k * MASK_H * MASK_W;
        for (int p = 0; p < MASK_H * MASK_W; p++)
            mask.at<float>(p / MASK_W, p % MASK_W) += coeff * proto[p];
    }
    for (int r = 0; r < MASK_H; r++)
        for (int c = 0; c < MASK_W; c++)
            mask.at<float>(r, c) = sigmoid(mask.at<float>(r, c));

    cv::Rect box = det.box;
    int mx1 = std::max(0,     (int)((box.x * scale + padW) / INPUT_W * MASK_W));
    int my1 = std::max(0,     (int)((box.y * scale + padH) / INPUT_H * MASK_H));
    int mx2 = std::min(MASK_W,(int)(((box.x + box.width)  * scale + padW) / INPUT_W * MASK_W));
    int my2 = std::min(MASK_H,(int)(((box.y + box.height) * scale + padH) / INPUT_H * MASK_H));

    if (mx2 <= mx1 || my2 <= my1) return;

    cv::Mat maskCrop = mask(cv::Rect(mx1, my1, mx2 - mx1, my2 - my1));
    cv::Mat maskResized;
    cv::resize(maskCrop, maskResized, cv::Size(box.width, box.height));

    cv::Mat roi = result(box);
    for (int r = 0; r < roi.rows; r++) {
        for (int c = 0; c < roi.cols; c++) {
            if (maskResized.at<float>(r, c) > 0.5f) {
                cv::Vec3b& px = roi.at<cv::Vec3b>(r, c);
                px[0] = (uchar)(px[0] * 0.5 + color[0] * 0.5);
                px[1] = (uchar)(px[1] * 0.5 + color[1] * 0.5);
                px[2] = (uchar)(px[2] * 0.5 + color[2] * 0.5);
            }
        }
    }
}

void drawBox(cv::Mat& img, const Detection& det)
{
    cv::Scalar color = CLASS_COLORS[det.classId % CLASS_COLORS.size()];
    cv::rectangle(img, det.box, color, 2);

    std::string label = CLASS_NAMES[det.classId]
                      + " " + std::to_string((int)(det.confidence * 100)) + "%";

    int baseline = 0;
    cv::Size ts  = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
    cv::Point tl = det.box.tl();

    cv::rectangle(img,
                  tl + cv::Point(0, -ts.height - 8),
                  tl + cv::Point(ts.width, 0),
                  color, cv::FILLED);
    cv::putText(img, label, tl + cv::Point(0, -4),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    // ── Usage ─────────────────────────────────────────────────────────────────
    if (argc < 4) {
        std::cerr
            << "Usage: " << argv[0]
            << " <mode> <model.onnx> <image.jpg> [output.jpg]\n\n"
            << "  mode:\n"
            << "    cuda  — CUDA EP, FP32  (fast startup, no engine build)\n"
            << "    fp16  — TensorRT EP, FP16  (~2x vs cuda; builds & caches engine on first run)\n"
            << "    int8  — TensorRT EP, INT8  (~4x vs cuda; needs ./trt_cache/calibration.flatbuffers)\n\n"
            << "  Example:\n"
            << "    " << argv[0] << " cuda  model.onnx culvert.jpg out_cuda.jpg\n"
            << "    " << argv[0] << " fp16  model.onnx culvert.jpg out_fp16.jpg\n"
            << "    " << argv[0] << " int8  model.onnx culvert.jpg out_int8.jpg\n";
        return 1;
    }

    const std::string mode       = argv[1];
    const std::string modelPath  = argv[2];
    const std::string imagePath  = argv[3];
    const std::string outputPath = (argc >= 5) ? argv[4] : "output.jpg";

    if (mode != "cuda" && mode != "fp16" && mode != "int8") {
        std::cerr << "Error: unknown mode '" << mode
                  << "'. Valid options: cuda | fp16 | int8\n";
        return 1;
    }

    // INT8 needs a pre-built calibration table — warn early rather than crash late
    if (mode == "int8") {
        const std::string calPath = "./trt_cache/calibration.flatbuffers";
        if (!std::filesystem::exists(calPath)) {
            std::cerr
                << "Warning: INT8 calibration table not found at " << calPath << "\n"
                << "  TensorRT will attempt to calibrate with random data — accuracy may be poor.\n"
                << "  Generate it with:\n"
                << "    trtexec --onnx=model.onnx --int8 --saveEngine=./trt_cache/engine.trt \\\n"
                << "            --calib=./trt_cache/calibration.flatbuffers\n\n";
        }
    }

    TimingReport t;
    t.mode = mode;
    auto t_total_start = now();

    // ── 1. Load image ─────────────────────────────────────────────────────────
    auto t0 = now();
    cv::Mat img = cv::imread(imagePath);
    t.imageLoad = ms(t0, now());

    if (img.empty()) {
        std::cerr << "Error: could not load image: " << imagePath << "\n";
        return 1;
    }
    int origW = img.cols, origH = img.rows;
    std::cout << "Image: " << origW << "x" << origH << "  mode: " << mode << "\n";

    // ── 2. Preprocess ─────────────────────────────────────────────────────────
    t0 = now();
    float scale; int padW, padH;
    std::vector<float> inputData = preprocess(img, scale, padW, padH);
    t.preprocess = ms(t0, now());

    // ── 3. ONNX Runtime session ───────────────────────────────────────────────
    t0 = now();
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_seg");
    Ort::SessionOptions sessionOpts;
    sessionOpts.SetIntraOpNumThreads(4);
    sessionOpts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // ── Mode-specific execution provider setup ────────────────────────────────
    if (mode == "cuda") {
        // ── CUDA EP — FP32 ────────────────────────────────────────────────────
        // Plain cuDNN inference. Fast startup, no engine compilation.
        OrtCUDAProviderOptions cudaOpts{};
        cudaOpts.device_id = 0;
        sessionOpts.AppendExecutionProvider_CUDA(cudaOpts);

    } else if (mode == "fp16") {
        // ── TensorRT EP — FP16 ───────────────────────────────────────────────
        // First run compiles and caches a fused FP16 engine (~30 s).
        // Subsequent runs load from cache and are ~2x faster than cuda mode.
        OrtTensorRTProviderOptions trtOpts{};
        trtOpts.device_id             = 0;
        trtOpts.trt_fp16_enable       = 1;        // FP16 kernel fusion
        trtOpts.trt_engine_cache_enable = 1;      // persist compiled engine
        trtOpts.trt_engine_cache_path = "./trt_cache";
        sessionOpts.AppendExecutionProvider_TensorRT(trtOpts);

        // CUDA fallback for any ops TRT doesn't support (e.g. custom YOLO heads)
        OrtCUDAProviderOptions cudaFallback{};
        cudaFallback.device_id = 0;
        sessionOpts.AppendExecutionProvider_CUDA(cudaFallback);

    } else if (mode == "int8") {
        // ── TensorRT EP — INT8 ───────────────────────────────────────────────
        // Fastest mode (~4x vs cuda). Requires calibration table generated from
        // representative culvert images so TRT knows the activation value ranges.
        //
        // Generate calibration table once with:
        //   trtexec --onnx=model.onnx --int8 \
        //           --calib=./trt_cache/calibration.flatbuffers
        OrtTensorRTProviderOptions trtOpts{};
        trtOpts.device_id                        = 0;
        trtOpts.trt_int8_enable                  = 1;
        trtOpts.trt_int8_calibration_table_name  = "calibration.flatbuffers";
        trtOpts.trt_engine_cache_enable          = 1;
        trtOpts.trt_engine_cache_path            = "./trt_cache";
        sessionOpts.AppendExecutionProvider_TensorRT(trtOpts);

        OrtCUDAProviderOptions cudaFallback{};
        cudaFallback.device_id = 0;
        sessionOpts.AppendExecutionProvider_CUDA(cudaFallback);
    }

    Ort::Session session(env, modelPath.c_str(), sessionOpts);
    t.sessionInit = ms(t0, now());

    // ── 4. Build input tensor ─────────────────────────────────────────────────
    std::vector<int64_t> inputShape = {1, 3, INPUT_H, INPUT_W};
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        inputData.data(), inputData.size(),
        inputShape.data(), inputShape.size()
    );

    // ── 5. Inference ──────────────────────────────────────────────────────────
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

    t0 = now();
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        inputNames, &inputTensor, 1,
        outputNames, 2
    );
    t.inference = ms(t0, now());

    float* out0Data = outputs[0].GetTensorMutableData<float>();
    float* out1Data = outputs[1].GetTensorMutableData<float>();

    // ── 6. Post-process ───────────────────────────────────────────────────────
    t0 = now();
    auto detections = postprocess(out0Data, scale, padW, padH, origW, origH);
    t.postprocess = ms(t0, now());

    std::cout << "Detections: " << detections.size() << "\n";
    for (const auto& d : detections)
        std::cout << "  " << CLASS_NAMES[d.classId]
                  << " (" << (int)(d.confidence * 100) << "%)"
                  << "  box=[" << d.box.x << "," << d.box.y
                  << "," << d.box.width << "," << d.box.height << "]\n";

    // ── 7. Draw masks ─────────────────────────────────────────────────────────
    cv::Mat result = img.clone();
    t0 = now();
    for (const auto& d : detections)
        applyMask(result, d, out1Data, scale, padW, padH, origW, origH);
    t.maskDraw = ms(t0, now());

    // ── 8. Draw boxes ─────────────────────────────────────────────────────────
    t0 = now();
    for (const auto& d : detections)
        drawBox(result, d);
    t.boxDraw = ms(t0, now());

    // ── 9. Save ───────────────────────────────────────────────────────────────
    t0 = now();
    cv::imwrite(outputPath, result);
    t.fileSave = ms(t0, now());

    t.total = ms(t_total_start, now());
    std::cout << "Saved → " << outputPath << "\n";

    // ── 10. Timing report ─────────────────────────────────────────────────────
    t.print();

    // Comment out if running headless / over SSH
    cv::imshow("YOLOv8 Seg — Culvert Detection [" + mode + "]", result);
    cv::waitKey(0);

    return 0;
}