//
//  main.cpp
//  mac_cam_ssd
//
//  Created by Weixing Zhang on 02/01/2019.
//  Copyright © 2019 Weixing Zhang. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui/highgui.hpp>

#include <mxnet/c_predict_api.h>
#include <mxnet/c_api.h>
//#include <mxnet-cpp/MxNetCpp.h>

const int FPS = 20;
const float threshold = 0.5;

const mx_float DEFAULT_MEAN = 110;
const mx_float DEFAULT_STD = 58;

// voc classes
//const std::string CATEGORY_NAMES[20] = {
//    "aeroplane", "bicycle", "bird", "boat",
//    "bottle", "bus", "car", "cat", "chair",
//    "cow", "diningtable", "dog", "horse",
//    "motorbike", "person", "pottedplant",
//    "sheep", "sofa", "train", "tvmonitor"};
// coco classes
const std::string CATEGORY_NAMES[80] = {
    "person", "bicycle", "car", "motorbike",
    "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "sofa", "pottedplant",
    "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

cv::Scalar LineColor = cvScalar(0,0,255); //overlay color
// cv::FONT_HERSHEY_SIMPLEX

// Read file to buffer
class BufferFile {
    public :
    std::string file_path_;
    int length_;
    char* buffer_;

    explicit BufferFile(std::string file_path)
    :file_path_(file_path) {

        std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            length_ = 0;
            buffer_ = NULL;
            return;
        }

        ifs.seekg(0, std::ios::end);
        length_ = static_cast<int>(ifs.tellg());
        ifs.seekg(0, std::ios::beg);
        std::cout << file_path.c_str() << " ... "<< length_ << " bytes\n";

        buffer_ = new char[sizeof(char) * length_];
        ifs.read(buffer_, length_);
        ifs.close();
    }

    int GetLength() {
        return length_;
    }
    char* GetBuffer() {
        return buffer_;
    }

    ~BufferFile() {
        if (buffer_) {
            delete[] buffer_;
            buffer_ = NULL;
        }
    }
};

// Channel conversion helper function
void CHW_to_HWC(mx_float* image_data, mx_float* new_image,
                const int channel, const int height, const int width)
{
    for (int i=0;i<height;++i)
    {
        for (int j=0;j<width;++j)
        {
            for (int k=0;k<channel;++k)
            {
                *new_image++=*(image_data+(channel-1-k)*height*width+j+i*width);
            }
        }
    }
}

// Get filtered output
void Filter_output(std::vector<float> cls, std::vector<float> score,
                   std::vector<float> coord,
                   std::string* out_cls,
                   float* out_xy)
{
    for (std::vector<float>::iterator it=score.begin();it!=score.end();++it)
    {
        if (*it>threshold)
        {
            int i=static_cast<int>(it-score.begin());
            *out_cls++=CATEGORY_NAMES[static_cast<int>(cls[i])];
            *out_xy++=coord[4*i];
            *out_xy++=coord[4*i+1];
            *out_xy++=coord[4*i+2];
            *out_xy++=coord[4*i+3];
            
        }
    }
}

int main(int argc, char **argv)
{
    //    std::string test_file;
    //    test_file = std::string("./test_legacy/000003.jpg");
    
    // Enter model path here
    std::string json_file = "./model/ssd512_coco-symbol.json";
    std::string param_file = "./model/ssd512_coco-0000.params";
    
    // CHW by default
    bool usingHWC = false;
    
    // Image size and channels
    int width = 512;
    int height = 512;
    int channels = 3;
    
    // MXPred parameters
    int dev_type = 1;  // 1: cpu, 2: gpu
    int dev_id = 0;
    mx_uint num_input_nodes = 1;  // 1 for feedforward
    const char* input_key[1] = {"data"};
    const char** input_keys = input_key;
    const mx_uint input_shape_indptr[2] = { 0, 4 };
    mx_uint input_shape_data[4]={1,1,1,1};
    if (usingHWC)
    {
        input_shape_data[1]=static_cast<mx_uint>(height);
        input_shape_data[2]=static_cast<mx_uint>(width);
        input_shape_data[3]=static_cast<mx_uint>(channels);
    }
    else
    {
        input_shape_data[1]=static_cast<mx_uint>(channels);
        input_shape_data[2]=static_cast<mx_uint>(height);
        input_shape_data[3]=static_cast<mx_uint>(width);
    }
    PredictorHandle pred_hnd;
    
    clock_t startTime,endTime;

    // Load model files
    startTime = clock();
    BufferFile json_data(json_file);
    BufferFile param_data(param_file);
    endTime = clock();
    
    double read_model_costs = (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000;
    std::cout << "load model file to memory costs : " << read_model_costs << "ms" << std::endl;
    
    // Initialize model
    startTime = clock();

    if (json_data.GetLength() == 0 ||
        param_data.GetLength() == 0) {
        return -1;
    }
    
    std::ifstream json_handle(json_file, std::ios::ate);
    std::string json_buffer;
    json_buffer.reserve(json_handle.tellg());
    json_handle.seekg(0, std::ios::beg);
    json_buffer.assign((std::istreambuf_iterator<char>(json_handle)), std::istreambuf_iterator<char>());
    
    assert(0==MXPredCreate(json_buffer.c_str(),
                           (const char*)param_data.GetBuffer(),
                           static_cast<int>(param_data.GetLength()),
                           dev_type,
                           dev_id,
                           num_input_nodes,
                           input_keys,
                           input_shape_indptr,
                           input_shape_data,
                           &pred_hnd));
    assert(pred_hnd);
    endTime = clock();
    
    double init_model_costs = (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000;
    std::cout << "init model costs : " << init_model_costs << "ms" << std::endl;

    // Initiate video capturing
    cv::VideoCapture cap;
    cap.open(0);
    
    if(!cap.isOpened())
    {
        std::cerr << "***Could not initialize capturing...***\n";
        std::cerr << "Current parameter's value: \n";
        return -1;
    }

    cv::namedWindow("new_window", cv::WINDOW_FREERATIO | cv::WINDOW_GUI_EXPANDED);
    cv::moveWindow("new_window", 1280/2-512/2, 800/2-512/2);
    
    // Get image stream
    cv::Mat frame;
    
    while(1)
    {
        // Read from cam
        cap >> frame;
        
        if(frame.empty())
        {
            std::cerr<<"frame is empty"<<std::endl;
            break;
        }
        
        int image_size = width * height * channels;
        cv::Mat im_ori = frame;
        //cv::imread(test_file, cv::IMREAD_COLOR);

        if (im_ori.empty())
        {
            std::cerr << "Can't get image. Please check cam. \n";
            assert(false);
        }
        
        // Resize to fit model
        cv::Mat im;
        resize(im_ori, im, cv::Size(width, height));
        
        // Construct image tensor
        std::vector<mx_float> image_data(image_size);
        
        // De-interleave and normalize
        // b g r ?
        unsigned char *ptr = im.ptr();
        float *data_ptr = image_data.data();
        float mean_b, mean_g, mean_r;
        float std_b, std_g, std_r;
        mean_b = mean_g = mean_r = DEFAULT_MEAN;
        std_b = std_g = std_r = DEFAULT_STD;
        
        for (int i = 0; i < image_size; i +=3) {
            *(data_ptr++) = (static_cast<float>(ptr[i]) - mean_r)/DEFAULT_STD;
        }
        for (int i = 1; i < image_size; i +=3) {
            *(data_ptr++) = (static_cast<float>(ptr[i]) - mean_g)/DEFAULT_STD;
        }
        for (int i = 2; i < image_size; i +=3) {
            *(data_ptr++) = (static_cast<float>(ptr[i]) - mean_b)/DEFAULT_STD;
        }
        
        // Swap axes if needed
        std::vector<mx_float> new_image(image_size);
        if (usingHWC)
        {
            CHW_to_HWC(image_data.data(), new_image.data(), channels, height, width);
        }
        else
        {
            new_image.clear();
        }
        
        // Bind tensor to model
        MXPredSetInput(pred_hnd, "data", image_data.data(), static_cast<mx_uint>(image_size));
        
        // Forward net
        MXPredForward(pred_hnd);
        
        // Get output shape before retrieving output
        mx_uint output_index = 0; // class id
        mx_uint *shape = NULL;
        mx_uint shape_len = 0;
        MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);
        mx_uint size = 1;
        for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];
        
        // Get network output
        std::vector<float> output_data_0(size);
        // MXNDArrayWaitToRead(&(output_data_0[0]));
        MXPredGetOutput(pred_hnd, output_index, output_data_0.data(), size);
        
        // repeat for output 1,2
        output_index = 1; // score
        *shape = NULL;
        shape_len = 0;
        MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);
        size = 1;
        for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];
        std::vector<float> output_data_1(size);
        MXPredGetOutput(pred_hnd, output_index, output_data_1.data(), size);
        
        output_index = 2; // xmin, ymin, xmax, ymax
        *shape = NULL;
        shape_len = 0;
        MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);
        size = 1;
        for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];
        std::vector<float> output_data_2(size);
        MXPredGetOutput(pred_hnd, output_index, output_data_2.data(), size);
        
        // Visualize prediction
        for (std::vector<float>::iterator it=output_data_1.begin();it!=output_data_1.end();++it)
        {
            if (*it>threshold)
            {
                int i=static_cast<int>(it-output_data_1.begin());
                cv::rectangle(im,
                              cvPoint(output_data_2[4*i], output_data_2[4*i+1]),
                              cvPoint(output_data_2[4*i+2], output_data_2[4*i+3]),
                              LineColor, 2);
                cv::putText(im,
                            CATEGORY_NAMES[static_cast<int>(output_data_0[i])],
                            cvPoint(output_data_2[4*i]+5, output_data_2[4*i+1]+25),
                            cv::FONT_HERSHEY_SIMPLEX, 1, LineColor);
            }
        }
        
        // Plot overlayed image
        cv::imshow("new_window", im);
        cv::resizeWindow("new_window", width, height);
        
        // Wait for any key to exit
        auto kin=cv::waitKey(1000/FPS);
        if (kin!=-1)
        {std::cout<<kin<<"...keyboard interrupted"<<std::endl; break;}
    }
    
    // Release Predictor
    MXPredFree(pred_hnd);
    std::cout << "free models" << std::endl;
    
    return 1;
}
