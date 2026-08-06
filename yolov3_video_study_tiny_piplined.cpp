#include <algorithm>
#include <array>
#include <vector>
#include <atomic>
#include <queue>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <zconf.h>
#include <thread>
#include <sys/stat.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>

// new include file related to thread handlig
#include <future>
#include <condition_variable>
#include <utility>

// next include files from VART program
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <chrono>
#include <xir/graph/graph.hpp>
#include "vitis/ai/collection_helper.hpp"
#include "common.h"
#include "utils.h"

#include <sstream>
#include <stdexcept>

using namespace std;
using namespace cv;
using namespace std::chrono;

using Clock = std::chrono::system_clock;

bool Lbox_on = false;

chrono::system_clock::time_point start_time, end_time, pre_end_time, dpu_end_time;

int idxInputImage = 0; // frame index of input video
int idxShowImage = 0;  // next frame index to be displayed
bool bReading = true;  // flag of reding input frame

std::mutex log_mutex;
constexpr int debug_period = 30;

GraphInfo shapes;

// 各スレッドの並列度
constexpr size_t kYoloOutputCount = 2;
constexpr size_t kPreprocessThreadCount = 2;
constexpr size_t kDpuThreadCount = 1;
constexpr size_t kPostprocessThreadCount = 1;
TensorShape inshapes[1];
TensorShape outshapes[kYoloOutputCount];

// デバッグ用に1フレームごとの処理時間を保存する構造体
struct FrameTimings
{
    int64_t preprocess_frame_us = 0;
    int64_t run_dpu_us = 0;
    int64_t postprocess_frame_us = 0;
    int64_t postprocess_us = 0;
    int64_t display_frame_us = 0;
};

// 表示するフレームのインデックスとフレーム画像,デバッグ情報を保持する構造体
struct imagePair
{
    int first = 0;
    Mat second;
    FrameTimings timings;

    imagePair() = default;
    imagePair(int index, Mat frame) : first(index), second(std::move(frame)) {}
};

int64_t elapsed_us(const Clock::time_point &start, const Clock::time_point &end)
{
    return duration_cast<microseconds>(end - start).count();
}

double us_to_ms(int64_t us)
{
    return static_cast<double>(us) / 1000.0;
}

bool should_print_debug(int count)
{
    return debug_period > 0 && (count == 1 || count % debug_period == 0);
}

class paircomp
{
public:
    bool operator()(const imagePair &n1, const imagePair &n2) const
    {
        if (n1.first == n2.first)
        {
            return (n1.first > n2.first);
        }

        return n1.first > n2.first;
    }
};

// Dpuへの入出力の構造体.
struct DpuInputFrame
{
    int index;
    Mat frame;
    vector<int8_t> input;
    FrameTimings timings;
};

struct DpuOutputFrame
{
    int index;
    Mat frame;
    std::array<vector<int8_t>, kYoloOutputCount> output;
    FrameTimings timings;
};

// input frames queue
queue<pair<int, Mat>> queueInput; // queue of FIFO
// display frames queue
priority_queue<imagePair, vector<imagePair>, paircomp> queueShow; // priority queue by index comp.

// ----------------------------------------------------------------------------------------
// キュー
// ----------------------------------------------------------------------------------------

// 複数スレッドからでも安全に使えるキュー
template <typename T>
class concurrent_queue
{
public:
    typedef typename std::queue<T>::size_type size_type;

private:
    std::queue<T> queue_;
    size_type capacity_;

    std::mutex mtx_;
    std::condition_variable can_pop_;
    std::condition_variable can_push_;

public:
    concurrent_queue(size_type capacity) : capacity_(capacity)
    {
        if (capacity_ == 0)
        {
            throw std::invalid_argument("capacity cannot be zero");
        }
    }

    void push(const T &value)
    {
        std::unique_lock<std::mutex> guard(mtx_);
        // wait 'can set'
        can_push_.wait(guard, [this]()
                       { return queue_.size() < capacity_; });
        queue_.push(value);
        // notify 'can get'
        can_pop_.notify_one();
    }

    void push(T &&value)
    {
        std::unique_lock<std::mutex> guard(mtx_);
        // wait 'can set'
        can_push_.wait(guard, [this]()
                       { return queue_.size() < capacity_; });
        queue_.push(std::move(value));
        // notify 'can get'
        can_pop_.notify_one();
    }

    // キューが満杯の場合は一番古い要素を削除して新しい要素を追加する
    void push_drop_oldest(T value)
    {
        std::unique_lock<std::mutex> guard(mtx_);
        if (queue_.size() >= capacity_)
        {
            queue_.pop();
        }
        queue_.push(std::move(value));
        can_pop_.notify_one();
    }

    T pop()
    {
        std::unique_lock<std::mutex> guard(mtx_);
        // wait 'can get'
        can_pop_.wait(guard, [this]()
                      { return !queue_.empty(); });
        T value = std::move(queue_.front());
        queue_.pop();
        // notify 'can set'
        can_push_.notify_one();
        return value;
    }

    // サイズの取得
    size_type size()
    {
        std::unique_lock<std::mutex> guard(mtx_);
        return queue_.size();
    }
};

// ----------------------------------------------------------------------------------------
// 動画を処理の前にdramにロードする関数
// ----------------------------------------------------------------------------------------
vector<Mat> preloadFramesToDram(const char *fileName)
{
    // 動画ファイルを開く.
    VideoCapture video;
    string videoFile = fileName;
    if (!video.open(videoFile))
    {
        cout << "Fail to open specified video file:" << videoFile << endl;
        exit(-1);
    }

    // 総フレーム数を取得し,framesの容量を確保.
    const auto frameCount = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));
    vector<Mat> frames;
    if (frameCount > 0)
    {
        frames.reserve(frameCount);
    }

    size_t totalBytes = 0;
    auto preload_start_time = Clock::now();

    // 1フレームずつデコードして,framesに格納.
    while (true)
    {
        Mat img;
        if (!video.read(img))
        {
            break;
        }

        totalBytes += img.total() * img.elemSize();
        frames.push_back(std::move(img));
    }

    video.release();
    auto preload_end_time = Clock::now();
    if (frames.empty())
    {
        cout << "No frames decoded from specified video file:" << videoFile << endl;
        exit(-1);
    }

    std::unique_lock<std::mutex> guard(log_mutex);
    cerr << fixed << setprecision(3)
         << "[preload] frames=" << frames.size()
         << " bytes=" << totalBytes
         << " time=" << us_to_ms(elapsed_us(preload_start_time, preload_end_time)) << "ms"
         << endl;
    return frames;
}

// -------------------------------------------------------------------------------------------------------
// 前処理
// -------------------------------------------------------------------------------------------------------

// void setInputImageForYOLO(const Mat &frame, int8_t *data,
//                           float input_scale)
// {
//     Mat img_copy;
//     int width = shapes.inTensorList[0].width;
//     int height = shapes.inTensorList[0].height;
//     int size = shapes.inTensorList[0].size;
//     image img_new = load_image_cv(frame);
//     image img_yolo = letterbox_image(img_new, width, height);

//     vector<float> bb(size);
//     for (int b = 0; b < height; ++b)
//     {
//         for (int c = 0; c < width; ++c)
//         {
//             for (int a = 0; a < 3; ++a)
//             {
//                 bb[b * width * 3 + c * 3 + a] =
//                     img_yolo.data[a * height * width + b * width + c];
//             }
//         }
//     }

//     float scale = pow(2, 7);
//     for (int i = 0; i < size; ++i)
//     {
//         data[i] = (int8_t)(bb.data()[i] * input_scale);
//         if (data[i] < 0)
//             data[i] = (int8_t)((float)(127 / scale) * input_scale);
//     }
//     free_image(img_new);
//     free_image(img_yolo);
// }

// Mat画像をDPUが受け取るin8_t型に変換する関数.
// preprocessの中身.
// void setInputPointer(const Mat &frame, int8_t *data,
//                      float scale)
// {
//     // 入力したxmodelのtensorのサイズを取得.
//     int width = shapes.inTensorList[0].width;
//     int height = shapes.inTensorList[0].height;
//     int size = shapes.inTensorList[0].size;

//     // 表示用にコピー.
//     Mat img = frame.clone();

//     // BGR=>RGBに変換して,リサイズする.
//     cvtColor(img, img, cv::COLOR_BGR2RGB);
//     Mat image2 = cv::Mat(height, width, CV_8SC3); // CV_8SC3 means 3ch singed char data type
//     cv::resize(img, image2, Size(width, height), 0, 0, cv::INTER_LINEAR);

//     // 画像の各pixelをint8_t型に変換する.
//     unsigned char *imdata = image2.data;
//     for (int i = 0; i < size; ++i)
//     {
//         float dataf = static_cast<float>(imdata[i]);
//         data[i] = static_cast<int8_t>(dataf * scale / 256.0f);
//         if (data[i] < 0)
//             data[i] = 127;
//     }
// }

void setInputPointer(const Mat &frame, int8_t *data)
{
    const int width = shapes.inTensorList[0].width;
    const int height = shapes.inTensorList[0].height;
    const int size = shapes.inTensorList[0].size;

    Mat resized;
    cv::resize(
        frame,
        resized,
        Size(width, height),
        0,
        0,
        cv::INTER_NEAREST);

    const uint8_t *src = resized.data;

    // resizedはBGR順、DPU入力はRGB順。
    // input_scale=64なので、pixel * 64 / 256 = pixel >> 2。
    for (int i = 0; i < size; i += 3)
    {
        data[i + 0] = static_cast<int8_t>(src[i + 2] >> 2); // R
        data[i + 1] = static_cast<int8_t>(src[i + 1] >> 2); // G
        data[i + 2] = static_cast<int8_t>(src[i + 0] >> 2); // B
    }
}

// Mat画像をDPUが受け取るin8_t型に変換する関数.
void preprocess(
    const Mat &frame,
    int8_t *input_data)
{
    if (Lbox_on)
    {
        // setInputImageForYOLO(frame, input_data);
    }
    else
    {
        // 基本的にこっちを使用する
        setInputPointer(frame, input_data);
    }
}

// preprocessのスレッド関数.
void preprocessFrame(
    const vector<Mat> &frames,
    std::atomic<int> &nextFrameIndex,
    concurrent_queue<DpuInputFrame> &out,
    int input_size)
{
    while (true)
    {
        // 複数のpreprocessスレッドが同時に実行されるので,atomicに次の処理するフレームのインデックスを取得する.
        const int index = nextFrameIndex.fetch_add(1);
        // 画像の取得.
        const Mat &frame = frames[static_cast<size_t>(index) % frames.size()];
        // 開始時間の記録.
        auto preprocess_start_time = Clock::now();

        // 取得した画像からdpuへ渡すための構造体の作成
        DpuInputFrame dpuInput;
        dpuInput.index = index;
        dpuInput.frame = frame;
        dpuInput.input.resize(input_size);

        // 処理本体.
        preprocess(dpuInput.frame, dpuInput.input.data());

        // 　デバッグ.
        auto preprocess_end_time = Clock::now();
        dpuInput.timings.preprocess_frame_us =
            elapsed_us(preprocess_start_time, preprocess_end_time);

        // dpuInのキューに追加.満タンなら待機.
        out.push(std::move(dpuInput));
    }
}

// -------------------------------------------------------------------------------------------------------
// DPU
// -------------------------------------------------------------------------------------------------------

template <typename TensorList>
void run_dpu(
    vart::Runner *runner,
    const TensorList &input_tensors,
    const TensorList &output_tensors,
    const vector<int> &output_mapping,
    int8_t *input_data,
    const std::array<int8_t *, kYoloOutputCount> &output_data)
{
    // DPU runnerに渡すためのTensorBufferを作成.
    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs;
    std::vector<std::unique_ptr<vart::TensorBuffer>> outputs;

    //
    std::vector<vart::TensorBuffer *> inputsPtr;
    std::vector<vart::TensorBuffer *> outputsPtr;

    // input_data を Tensorbufferに.
    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        input_data, input_tensors[0].get()));

    //
    for (size_t i = 0; i < output_data.size(); ++i)
    {
        outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
            output_data[i], output_tensors[output_mapping[i]].get()));
    }

    inputsPtr.push_back(inputs[0].get());
    for (auto &output : outputs)
    {
        outputsPtr.push_back(output.get());
    }

    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    runner->wait(job_id.first, -1);
}

// DPUのスレッド関数.
void DPUFrame(
    vart::Runner *runner,
    concurrent_queue<DpuInputFrame> &in,
    concurrent_queue<DpuOutputFrame> &out)
{
    auto inputTensors = cloneTensorBuffer(runner->get_input_tensors());
    auto outputTensors = cloneTensorBuffer(runner->get_output_tensors());
    vector<int> output_mapping = shapes.output_mapping;

    while (true)
    {
        // dpuInキューからDpuInputFrameを取得.
        auto dpuInput = in.pop();

        // 出力用の構造体の作成.
        DpuOutputFrame dpuOutput;
        dpuOutput.index = dpuInput.index;
        dpuOutput.frame = dpuInput.frame;
        dpuOutput.timings = dpuInput.timings;

        // 出力用のバッファの確保.tinyyolov3であれば出力は2本.
        for (size_t i = 0; i < dpuOutput.output.size(); ++i)
        {
            dpuOutput.output[i].resize(shapes.outTensorList[i].size);
        }

        // run_dpuに渡すためにouput_dataを作成し,run_dpuを実行
        std::array<int8_t *, kYoloOutputCount> output_data = {
            dpuOutput.output[0].data(),
            dpuOutput.output[1].data()};

        // ここ以前の処理は0.006msで終了したため,dpuのみを計測.
        auto run_dpu_start_time = Clock::now();

        run_dpu(
            runner,
            inputTensors,
            outputTensors,
            output_mapping,
            dpuInput.input.data(),
            output_data);

        // デバッグ用時間測定
        auto run_dpu_end_time = Clock::now();
        dpuOutput.timings.run_dpu_us =
            elapsed_us(run_dpu_start_time, run_dpu_end_time);

        // dpuOutのキューに追加.
        out.push(std::move(dpuOutput));
    }
}

// --------------------------------------------------------------------------------------------
// 後処理
// --------------------------------------------------------------------------------------------

// DPUの出力tensorを受取り,画像にbboxを描画する関数.
void postprocess(Mat &img, const vector<int8_t *> &out, const GraphInfo &shapes,
                 const float &scale, const int &sHeight, const int &sWidth)
{
    // sHeight and sWidth come from the xmodel input tensor.
    // 検出候補のbboxを格納する配列.
    vector<vector<float>> boxes;
    char fname[256];

    // dpuの出力tensorを1本ずつ取り出す,tinyyolov3であれば2本.
    for (size_t i = 0; i < out.size(); i++)
    {
        // 出力tensorの形を取得.
        int channel = shapes.outTensorList[i].channel;
        int width = shapes.outTensorList[i].width;
        int height = shapes.outTensorList[i].height;
        int sizeOut = shapes.outTensorList[i].size;

        // 検出候補のbboxを格納する配列を確保.
        vector<float> result(sizeOut);
        boxes.reserve(sizeOut);

        // DPUから出力されたoutからbbox候補を取り出してboxesに追加.
        detect(boxes, out[i], channel, height, width, i, sHeight, sWidth, scale);
    }

    /* Restore the correct coordinate frame of the original image */
    if (Lbox_on)
    {
        correct_region_boxes(boxes, boxes.size(), img.cols, img.rows, sWidth, sHeight);
    }

    // NMSにより,重なり過ぎているbboxを削除.
    vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

    // 実際の描画処理.
    float h = img.rows;
    float w = img.cols;
    for (size_t i = 0; i < res.size(); ++i)
    {
        // 非正規化
        float xmin = (res[i][0] - res[i][2] / 2.0) * w + 1.0;
        float ymin = (res[i][1] - res[i][3] / 2.0) * h + 1.0;
        float xmax = (res[i][0] + res[i][2] / 2.0) * w + 1.0;
        float ymax = (res[i][1] + res[i][3] / 2.0) * h + 1.0;

        // cout<<res[i][res[i][4] + 6]<<" "; // (res[i][4]=class#)+6(offset) means class_score due to the results of apply NMS.
        // cout<<res[i][4] << " "; // most confident class number
        // cout<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax<<endl;

        // 検出信頼度が閾値以上のbboxのみ描画する.
        if (res[i][res[i][4] + 6] > CONF)
        {
            int type = res[i][4];

            if (type == 0)
            {
                rectangle(img, Point(xmin, ymin), Point(xmax, ymax),
                          Scalar(0, 0, 255), 1, 1, 0);
            }
            else if (type == 1)
            {
                rectangle(img, Point(xmin, ymin), Point(xmax, ymax),
                          Scalar(255, 0, 0), 1, 1, 0);
            }
            else
            {
                rectangle(img, Point(xmin, ymin), Point(xmax, ymax),
                          Scalar(0, 255, 255), 1, 1, 0);
            }
        }
    }
}

void postprocessFrame(
    concurrent_queue<DpuOutputFrame> &in,
    concurrent_queue<imagePair> &out,
    float output_scale,
    int input_height,
    int input_width)
{
    while (true)
    {
        // dpuOutキューからDpuOutputFrameをpopする.
        auto dpuOutput = in.pop();
        auto postprocess_frame_start_time = Clock::now();
        Mat img = dpuOutput.frame.clone();

        vector<int8_t *> results = {
            dpuOutput.output[0].data(),
            dpuOutput.output[1].data()};
        auto postprocess_start_time = Clock::now();

        // 本体.
        postprocess(
            img,
            results,
            shapes,
            output_scale,
            input_height,
            input_width);
        auto postprocess_end_time = Clock::now();
        dpuOutput.timings.postprocess_us =
            elapsed_us(postprocess_start_time, postprocess_end_time);

        auto postprocess_frame_end_time = Clock::now();
        imagePair pair(dpuOutput.index, std::move(img));
        pair.timings = dpuOutput.timings;
        pair.timings.postprocess_frame_us =
            elapsed_us(postprocess_frame_start_time, postprocess_frame_end_time);

        // shwキューに追加.満タンなら一番古い要素を削除して新しい要素を追加する.
        out.push_drop_oldest(std::move(pair));
    }
}

// -------------------------------------------------------------------------------------------------------
// 動画を表示
// -------------------------------------------------------------------------------------------------------
void displayFrame(
    concurrent_queue<imagePair> &in,
    std::atomic<int> &nextFrameIndex,
    size_t preloadedFrameCount,
    concurrent_queue<DpuInputFrame> &dpuIn,
    concurrent_queue<DpuOutputFrame> &dpuOut)
{
    Mat frame;
    int index;
    int displayedCount = 0;
    while (true)
    {
        auto pairIndexImg = in.pop();
        frame = pairIndexImg.second;
        index = pairIndexImg.first;

        if (frame.rows <= 0 || frame.cols <= 0)
        {
            continue;
        }

        // FPSの計算,表示
        auto display_start_time = Clock::now();
        auto show_time = chrono::system_clock::now();
        auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
        stringstream buffer;
        // この評価式に変更は加えない.
        buffer << fixed << setprecision(1)
               << (float)pairIndexImg.first / (dura / 1000000.f);
        string a = buffer.str() + " FPS";
        putText(frame, a, cv::Point(10, 15), 1, 1, cv::Scalar{0, 0, 240}, 1);
        imshow("YOLOv3 Detection@Xilinx DPU", frame);

        auto key = waitKey(1);
        if (key == 27)
        {
            bReading = false; // usually true, set false only when 'q' key is pushed.
            exit(0);
        }

        // debug用時間計測.
        auto display_end_time = Clock::now();
        pairIndexImg.timings.display_frame_us =
            elapsed_us(display_start_time, display_end_time);

        // debug情報の表示
        // 各Frameの処理時間とキュー情報を同じタイミングで表示する.
        ++displayedCount;
        if (should_print_debug(displayedCount))
        {
            std::unique_lock<std::mutex> guard(log_mutex);
            cerr << fixed << setprecision(3)
                 << "[time] frame=" << index
                 << " preprocessFrame="
                 << us_to_ms(pairIndexImg.timings.preprocess_frame_us) << "ms"
                 << " DPUFrame=" << us_to_ms(pairIndexImg.timings.run_dpu_us) << "ms"
                 << " postprocessFrame="
                 << us_to_ms(pairIndexImg.timings.postprocess_frame_us) << "ms"
                 << " postprocess="
                 << us_to_ms(pairIndexImg.timings.postprocess_us) << "ms"
                 << " displayFrame="
                 << us_to_ms(pairIndexImg.timings.display_frame_us) << "ms"
                 << endl;
            cerr << "[queue] frame=" << index
                 << " sourceIndex=" << nextFrameIndex.load()
                 << " preloaded=" << preloadedFrameCount
                 << " dpuIn=" << dpuIn.size()
                 << " dpuOut=" << dpuOut.size()
                 << " shw=" << in.size()
                 << endl;
        }
    }
}

// -------------------------------------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------------------------------------

int main(const int argc, const char **argv)
{
    // 引数の数をチェック
    cout << "concurrency = " << std::thread::hardware_concurrency() << std::endl;
    // Check args
    if (argc != 3)
    {
        cout << "Usage of yolov3: ./resnet50 [model_file] [jpg image]" << endl;
        // #of images = batch size
        return -1;
    }
    auto xmodel_file = std::string(argv[1]);
    auto preloadedFrames = preloadFramesToDram(argv[2]);

    // dpu runnerの作成.
    auto graph = xir::Graph::deserialize(xmodel_file);
    auto subgraph = get_dpu_subgraph(graph.get());
    CHECK_EQ(subgraph.size(), 1u)
        << "yolov3 should have one and only one dpu subgraph." << endl;
    cout << "create running for subgraph: " << subgraph[0]->get_name() << endl;

    auto runner =
        vart::Runner::create_runner(subgraph[0], "run");
    auto inputTensors = runner->get_input_tensors();
    auto outputTensors = runner->get_output_tensors();

    //  xmodelの入出力のtensor情報を取得
    int inputCnt = inputTensors.size();
    int outputCnt = outputTensors.size();
    CHECK_EQ(outputCnt, static_cast<int>(kYoloOutputCount))
        << "yolov3-tiny should have two output tensors." << endl;

    //
    TensorShape inshapes[inputCnt];
    TensorShape outshapes[outputCnt];
    shapes.inTensorList = inshapes;
    shapes.outTensorList = outshapes; // get output size
    getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);
    CHECK_EQ(shapes.output_mapping.size(), kYoloOutputCount)
        << "yolov3-tiny output mapping should have two entries." << endl;

    // 前処理,後処理,DPU処理に必要なパラメータの取得.
    const int inHeight = shapes.inTensorList[0].height;
    const int inWidth = shapes.inTensorList[0].width;
    const int batchSize = 1; // fixed
    const int inSize = shapes.inTensorList[0].size * batchSize;

    // input scaleのサイズを取得.
    auto input_scale = get_input_scale(runner->get_input_tensors()[0]);
    cerr << "input_scale = "
         << input_scale
         << endl;
    // input scaleが64のモデルを前提としてpreprocessのコードを組んでいるため,
    // それ以外の値が設定されている場合はエラーとする.
    constexpr float expected_input_scale = 64.0f;
    constexpr float tolerance = 1.0e-4f;

    if (std::abs(input_scale - expected_input_scale) > tolerance)
    {
        throw std::runtime_error(
            "Unsupported input_scale: " +
            std::to_string(input_scale) +
            " (expected 64)");
    }

    vector<int> output_mapping = shapes.output_mapping;
    auto conf_output_scale =
        get_output_scale(runner->get_output_tensors()[output_mapping[1]]);

    {
        std::unique_lock<std::mutex> guard(log_mutex);
        cerr << "[debug] debug_period=" << debug_period
             << " preloadedFrames=" << preloadedFrames.size()
             << " timing/queue output: first displayed frame and every debug_period displayed frames"
             << endl;
    }

    std::atomic<int> nextFrameIndex{0};

    // 各種キューの作成
    concurrent_queue<imagePair> shw(100);
    concurrent_queue<DpuInputFrame> dpuIn(100);
    concurrent_queue<DpuOutputFrame> dpuOut(kPostprocessThreadCount);

    // スレッドの作成
    vector<thread> threadsList;
    threadsList.reserve(
        1 + kPreprocessThreadCount + kDpuThreadCount + kPostprocessThreadCount);
    start_time = chrono::system_clock::now();
    for (size_t i = 0; i < kPreprocessThreadCount; ++i)
    {
        threadsList.emplace_back(
            preprocessFrame,
            ref(preloadedFrames),
            ref(nextFrameIndex),
            ref(dpuIn),
            inSize);
    }
    for (size_t i = 0; i < kDpuThreadCount; ++i)
    {
        threadsList.emplace_back(DPUFrame, runner.get(), ref(dpuIn), ref(dpuOut));
    }
    for (size_t i = 0; i < kPostprocessThreadCount; ++i)
    {
        threadsList.emplace_back(
            postprocessFrame, ref(dpuOut), ref(shw), conf_output_scale, inHeight, inWidth);
    }
    threadsList.emplace_back(
        displayFrame,
        ref(shw),
        ref(nextFrameIndex),
        preloadedFrames.size(),
        ref(dpuIn),
        ref(dpuOut));

    for (auto &worker : threadsList)
    {
        worker.join();
    }

    return 0;
}
