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

using namespace std;
using namespace cv;
using namespace std::chrono;

bool Lbox_on = false;

chrono::system_clock::time_point start_time, end_time, pre_end_time, dpu_end_time;

int idxInputImage = 0; // frame index of input video
int idxShowImage = 0;  // next frame index to be displayed
bool bReading = true;  // flag of reding input frame

typedef pair<int, Mat> imagePair;
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

// input frames queue
queue<pair<int, Mat>> queueInput; // queue of FIFO
// display frames queue
priority_queue<imagePair, vector<imagePair>, paircomp> queueShow; // priority queue by index comp.

GraphInfo shapes;
TensorShape inshapes[1];
TensorShape outshapes[3];

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

    T pop()
    {
        std::unique_lock<std::mutex> guard(mtx_);
        // wait 'can get'
        can_pop_.wait(guard, [this]()
                      { return !queue_.empty(); });
        T value = queue_.front();
        queue_.pop();
        // notify 'can set'
        can_push_.notify_one();
        return value;
    }
    int size() { return queue_.size(); }
};

void readFrame(const char *fileName, concurrent_queue<imagePair> &out)
{
    static int loop = 3; // video end of three times play
    VideoCapture video;
    string videoFile = fileName;
    start_time = chrono::system_clock::now();

    while (loop > 0)
    {
        // loop--; //infinite loop
        if (!video.open(videoFile))
        {
            cout << "Fail to open specified video file:" << videoFile << endl;
            exit(-1);
        }

        while (true)
        {
            // usleep(20000); // No performanec increase if 20000 --> 2000
            // auto start_readtime = chrono::system_clock::now();
            Mat img;
            if (!video.read(img))
            {
                break;
            }

            auto pair = make_pair(idxInputImage++, img);
            end_time = chrono::system_clock::now();
            // cout << "\nread img time= " <<
            //	    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_readtime).count()
            //	    << " [mS]" << endl;
            out.push(pair);
            // cout << "index=" << idxInputImage << "\n" << flush;
            // cout << "q size=" << queueInput.size() << "\n" << flush;
        }

        video.release();
    }
    exit(0);
}

void displayFrame(concurrent_queue<imagePair> &in)
{
    Mat frame;
    int index;
    while (true)
    {
        auto pairIndexImg = in.pop();
        frame = pairIndexImg.second;
        index = pairIndexImg.first;
        if (frame.rows <= 0 || frame.cols <= 0)
        {
            continue;
        }

        auto show_time = chrono::system_clock::now();
        auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
        stringstream buffer;
        buffer << fixed << setprecision(1)
               << (float)pairIndexImg.first / (dura / 1000000.f);
        string a = buffer.str() + " FPS";
        putText(frame, a, cv::Point(10, 15), 1, 1, cv::Scalar{0, 0, 240}, 1);
        // cout << "FPS=" << buffer.str() << "\n" << flush;
        // cout << "index = " << index << flush;
        // if (index%2) {
        imshow("YOLOv3 Detection@Xilinx DPU", frame);
        auto key = waitKey(1);
        if (key == 27)
        {
            bReading = false; // usually true, set false only when 'q' key is pushed.
            exit(0);
        }
        //}
        auto disp_end_time = chrono::system_clock::now();
        // cout << "\ndisplay time= " <<
        //	    std::chrono::duration_cast<std::chrono::milliseconds>(disp_end_time - show_time).count()
        //      << " [mS]" << flush;
    }
}

// for debug
void write_output(const string &name, const int8_t *result, const int &size0)
{
    ofstream ofs(name, ios_base::binary);
    if (!ofs)
    {
        cout << "Cannot open output file!!" << endl;
    }
    else
    {
        ofs.write((char *)&result[0], sizeof(int8_t) * size0);
    }
}

void post_process(Mat &img, const vector<int8_t *> &out, const GraphInfo &shapes,
                  const float &scale, const int &sHeight, const int &sWidth)
{
    // sHeight, sWidth = 416, 416
    vector<vector<float>> boxes;
    char fname[256];
    for (size_t i = 0; i < out.size(); i++)
    {
        int channel = shapes.outTensorList[i].channel;
        int width = shapes.outTensorList[i].width;
        int height = shapes.outTensorList[i].height;
        int sizeOut = shapes.outTensorList[i].size;
        // cout << "width, height = " << width << ", " << height << endl; // debug
        // cout << "channel, sizeOut = " << channel << ", " << sizeOut << endl; // debug

        vector<float> result(sizeOut);
        boxes.reserve(sizeOut);

        /* Store every output node results */
        // scaling_cast(out[i], scale, result);

        detect(boxes, out[i], channel, height, width, i, sHeight, sWidth, scale);

        /* debug
        sprintf(fname, "out%d.dat", i);
        cout << "binary file output : " << fname << endl;
        cout << "" << endl;

        write_binary(fname, result);
        */
    }

    /* Restore the correct coordinate frame of the original image */
    if (Lbox_on)
    {
        correct_region_boxes(boxes, boxes.size(), img.cols, img.rows, sWidth, sHeight);
    }
    /* Apply the computation for NMS */
    // cout << "boxes size: " << boxes.size() << endl; // debug
    vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

    // cout << "boxes size after NMS: " << res.size() << endl;
    // cout << "class conf, class, xmin, ymin, xmax, ymax" << endl;
    float h = img.rows;
    float w = img.cols;
    for (size_t i = 0; i < res.size(); ++i)
    {
        float xmin = (res[i][0] - res[i][2] / 2.0) * w + 1.0;
        float ymin = (res[i][1] - res[i][3] / 2.0) * h + 1.0;
        float xmax = (res[i][0] + res[i][2] / 2.0) * w + 1.0;
        float ymax = (res[i][1] + res[i][3] / 2.0) * h + 1.0;

        // cout<<res[i][res[i][4] + 6]<<" "; // (res[i][4]=class#)+6(offset) means class_score due to the results of apply NMS.
        // cout<<res[i][4] << " "; // most confident class number
        // cout<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax<<endl;

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

void setInputImageForYOLO(vart::Runner *runner, const Mat &frame, int8_t *data,
                          float input_scale)
{
    Mat img_copy;
    int width = shapes.inTensorList[0].width;
    int height = shapes.inTensorList[0].height;
    int size = shapes.inTensorList[0].size;
    image img_new = load_image_cv(frame);
    image img_yolo = letterbox_image(img_new, width, height);

    vector<float> bb(size);
    for (int b = 0; b < height; ++b)
    {
        for (int c = 0; c < width; ++c)
        {
            for (int a = 0; a < 3; ++a)
            {
                bb[b * width * 3 + c * 3 + a] =
                    img_yolo.data[a * height * width + b * width + c];
            }
        }
    }

    float scale = pow(2, 7);
    for (int i = 0; i < size; ++i)
    {
        data[i] = (int8_t)(bb.data()[i] * input_scale);
        if (data[i] < 0)
            data[i] = (int8_t)((float)(127 / scale) * input_scale);
    }
    free_image(img_new);
    free_image(img_yolo);
}

void setInputPointer(vart::Runner *runner, const Mat &frame, int8_t *data,
                     float scale)
{
    int width = shapes.inTensorList[0].width;
    int height = shapes.inTensorList[0].height;
    int size = shapes.inTensorList[0].size;

    Mat img = frame.clone();
    cvtColor(img, img, cv::COLOR_BGR2RGB);
    Mat image2 = cv::Mat(height, width, CV_8SC3); // CV_8SC3 means 3ch singed char data type
    cv::resize(img, image2, Size(width, height), 0, 0, cv::INTER_LINEAR);

    // unsigned char* imdata = img.data;
    unsigned char *imdata = image2.data;
    for (int i = 0; i < size; ++i)
    {
        float dataf = static_cast<float>(imdata[i]);
        data[i] = static_cast<int8_t>(dataf * scale / 256.0f);
        if (data[i] < 0)
            data[i] = 127;
    }
}

void preprocess(
    vart::Runner *runner,
    const Mat &frame,
    int8_t *input_data,
    float input_scale)
{
    if (Lbox_on)
    {
        setInputImageForYOLO(runner, frame, input_data, input_scale);
    }
    else
    {
        setInputPointer(runner, frame, input_data, input_scale);
    }
}

template <typename TensorList>
void run_dpu(
    vart::Runner *runner,
    const TensorList &input_tensors,
    const TensorList &output_tensors,
    const vector<int> &output_mapping,
    int8_t *input_data,
    const std::array<int8_t *, 3> &output_data)
{
    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs;
    std::vector<std::unique_ptr<vart::TensorBuffer>> outputs;
    std::vector<vart::TensorBuffer *> inputsPtr;
    std::vector<vart::TensorBuffer *> outputsPtr;

    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        input_data, input_tensors[0].get()));
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

Mat postprocess(
    const Mat &frame,
    const std::array<int8_t *, 3> &output_data,
    const GraphInfo &graph_info,
    float output_scale,
    int input_height,
    int input_width)
{
    Mat img = frame.clone();
    vector<int8_t *> results(output_data.begin(), output_data.end());
    post_process(img, results, graph_info, output_scale, input_height, input_width);
    return img;
}

void runYOLO(vart::Runner *runner, concurrent_queue<imagePair> &in, concurrent_queue<imagePair> &out)
{

    auto inputTensors = cloneTensorBuffer(runner->get_input_tensors());
    auto outputTensors = cloneTensorBuffer(runner->get_output_tensors());

    // set input pointer
    int inHeight = shapes.inTensorList[0].height;
    int inWidth = shapes.inTensorList[0].width;
    int inChannel = 3; // fixed
    int batchSize = 1; // fixed
    int inSize = inHeight * inWidth * inChannel;
    int8_t *imageInputs = new int8_t[inSize * batchSize];

    /*
    cout << "in_height = " << inHeight << endl;
    cout << "in_width = " << inWidth << endl;
    cout << "in_channel = " << inChannel << endl;
    cout << "batch size = " << batchSize << endl;
    cout << "\n";
    */
    // set output pointer
    vector<int> output_mapping = shapes.output_mapping;
    auto conf_output_scale =
        get_output_scale(runner->get_output_tensors()[output_mapping[1]]);

    // make output vector
    const int size0 = shapes.outTensorList[0].size;
    // cout << "size0 = " << size0 << endl; // debug
    int8_t *result0 = new int8_t[size0 * batchSize];
    const int size1 = shapes.outTensorList[1].size;
    // cout << "size1 = " << size1 << endl; // debug
    int8_t *result1 = new int8_t[size1 * batchSize];
    const int size2 = shapes.outTensorList[2].size;
    // cout << "size2 = " << size2 << endl; // debug
    int8_t *result2 = new int8_t[size2 * batchSize];
    std::array<int8_t *, 3> output_data = {result0, result1, result2};

    auto input_scale = get_input_scale(runner->get_input_tensors()[0]);

    while (true)
    {
        auto pairIndexImage = in.pop();
        auto yolo_start_time = std::chrono::system_clock::now(); // runYOLO starttime

        preprocess(runner, pairIndexImage.second, imageInputs, input_scale);
        // pre_end_time = std::chrono::system_clock::now();

        run_dpu(
            runner,
            inputTensors,
            outputTensors,
            output_mapping,
            imageInputs,
            output_data);
        // cout << "Done execution" << endl; // debug

        // dpu_end_time = std::chrono::system_clock::now();

        // check output
        /* debug
        write_output("out0.bin", result0, size0);
        write_output("out1.bin", result1, size1);
        write_output("out2.bin", result2, size2);
        */
        pairIndexImage.second = postprocess(
            pairIndexImage.second,
            output_data,
            shapes,
            conf_output_scale,
            inHeight,
            inWidth);
        // cv::imwrite("result.jpg", image2);
        out.push(pairIndexImage);
        auto yolo_end_time = std::chrono::system_clock::now();
        // cout << "\nrunYOLO time= " <<
        //    std::chrono::duration_cast<std::chrono::milliseconds>(yolo_end_time - yolo_start_time).count()
        //    << " [mS]" << endl;
    }
    delete[] imageInputs;
    delete[] result0;
    delete[] result1;
    delete[] result2;

    /*
    cout << "\npre_process time = " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(pre_end_time - start_time).count()
        << " [mS]" << endl;
    cout << "DPU time = " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(dpu_end_time - pre_end_time).count()
        << " [mS]" << endl;
    cout << "post_process time = " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - dpu_end_time).count()
        << " [mS]" << endl;
    cout << "-------------------------------------------" << endl;
    cout << "total proc. time = " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
        << " [mS]" << endl;

    cout << "\nDone yolov3." << endl;*/
}

int main(const int argc, const char **argv)
{
    cout << "concurrency = " << std::thread::hardware_concurrency() << std::endl;
    // Check args
    if (argc != 3)
    {
        cout << "Usage of yolov3: ./resnet50 [model_file] [jpg image]" << endl;
        // #of images = batch size
        return -1;
    }
    auto xmodel_file = std::string(argv[1]);

    // create dpu runner
    auto graph = xir::Graph::deserialize(xmodel_file);
    auto subgraph = get_dpu_subgraph(graph.get());
    CHECK_EQ(subgraph.size(), 1u)
        << "yolov3 should have one and only one dpu subgraph." << endl;
    cout << "create running for subgraph: " << subgraph[0]->get_name() << endl;

    auto attrs = xir::Attrs::create();
    auto runner =
        vart::Runner::create_runner(subgraph[0], "run");
    auto runner1 =
        vart::Runner::create_runner(subgraph[0], "run");
    auto runner2 =
        vart::Runner::create_runner(subgraph[0], "run");
    auto runner3 =
        vart::Runner::create_runner(subgraph[0], "run");
    // auto runner = vart::Runner::create_runner(subgraph[0], "run");
    // start_time = chrono::system_clock::now();
    auto inputTensors = runner->get_input_tensors();
    auto outputTensors = runner->get_output_tensors();

    // write_output("input.bin", imageInputs, inSize); // for debug
    //  get in/out tenosrs
    int inputCnt = inputTensors.size();
    int outputCnt = outputTensors.size();
    TensorShape inshapes[inputCnt];
    TensorShape outshapes[outputCnt];
    shapes.inTensorList = inshapes;
    shapes.outTensorList = outshapes; // get output size
    getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

    concurrent_queue<imagePair> fr(30), shw(30);
    array<thread, 4> threadsList = {
        thread(readFrame, argv[2], ref(fr)),
        thread(displayFrame, ref(shw)),
        thread(runYOLO, runner.get(), ref(fr), ref(shw)),
        thread(runYOLO, runner1.get(), ref(fr), ref(shw)),
    };

    for (int i = 0; i < 4; i++)
    {
        threadsList[i].join();
    }

    return 0;
}
