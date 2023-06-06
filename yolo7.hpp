#ifndef __YOLO7_HPP__
#define __YOLO7_HPP__ 
#include "utils.hpp"


#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.5
#define CLASSIC_MEM_
// #define UNIFIED_MEM_

class Yolo7
{
private:
    Logger gLogger;
    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context;
    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine;
    std::vector<void*> buffers;
    const int inputIndex = 0;
    const int outputIndex1 = 1;
    const int outputIndex2 = 2;
    const int outputIndex3 = 3;
    float *blob;
    float *prob1,*prob2,*prob3;
    cudaStream_t stream;
    int img_w;
    int img_h;
    cv::Mat re;
    float scale;
    size_t output_size1,output_size2,output_size3;

    float anchor_grid_w_1[3] = {12.0,19.0,40.0};
    float anchor_grid_h_1[3] = {16.0,36.0,28.0};

    float anchor_grid_w_2[3] = {36,76,72};
    float anchor_grid_h_2[3] = {75,55,146};

    float anchor_grid_w_3[3] = {142,192,459};
    float anchor_grid_h_3[3] = {110,243,401};

    void generete_proposal_scale(float* feat_blob, float prob_threshold, std::vector<Object>& objects,int nx, int ny,float stride,float * anchor_grid_w,float * anchor_grid_h);
    void decode_outputs(float* prob1,float* prob2,float* prob3, std::vector<Object>& objects, float scale, const int img_w, const int img_h);

public:
    Yolo7(std::string engine_file_path, int img_w, int img_h);
    ~Yolo7();
    void detect(const cv::Mat &img,std::vector<Object> &objects);

};


#endif
