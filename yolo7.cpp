#include "yolo7.hpp"

Yolo7::Yolo7(std::string engine_file_path,int img_w ,int img_h)
{


    // TODO: check for existance of engine file or crete engine...
    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) 
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    unique_ptr<IRuntime,TRTDestroy> runtime{createInferRuntime(gLogger)};
    assert(runtime != nullptr);
    engine.reset(runtime->deserializeCudaEngine(trtModelStream, size));
    assert(engine != nullptr); 
    context.reset( engine->createExecutionContext() );
    assert(context != nullptr);
    
    delete[] trtModelStream;
    buffers.reserve(engine->getNbBindings());
    cout<<"------------------------------"<<endl;
    for (size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * 1 * sizeof(float);
        // cudaMalloc(&buffers_base_q[i], binding_size);
        std::cout<<engine->getBindingName(i)<<std::endl;
    }

    assert(engine->getNbBindings() == 4); // it must be 4
    auto out_dims1 = engine->getBindingDimensions(1);
    auto out_dims2 = engine->getBindingDimensions(2);
    auto out_dims3 = engine->getBindingDimensions(3);

    output_size1 = getSizeByDim(out_dims1);
    output_size2 = getSizeByDim(out_dims2);
    output_size3 = getSizeByDim(out_dims3);
    cout<<output_size1<<" | "<<output_size2<<" | "<<output_size3<<"\n";

#ifdef CLASSIC_MEM_
    prob1 = new float[output_size1];
    prob2 = new float[output_size2];
    prob3 = new float[output_size3];

    blob = new float[INPUT_H*INPUT_W*3];

    CHECK(cudaMalloc(&buffers[inputIndex], 3 * INPUT_W * INPUT_H * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], output_size1*sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex2], output_size2*sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex3], output_size3*sizeof(float)));
#endif
#ifdef UNIFIED_MEM_
    cudaMallocManaged((void **)&blob,3 * INPUT_W * INPUT_H * sizeof(float),cudaMemAttachHost);
    cudaMallocManaged((void **)&prob1,output_size1*sizeof(float));
    cudaMallocManaged((void **)&prob2,output_size2*sizeof(float));
    cudaMallocManaged((void **)&prob3,output_size3*sizeof(float));

    buffers[inputIndex] = (void *) blob;
    buffers[outputIndex1] = (void *) prob1;
    buffers[outputIndex2] = (void *) prob2;
    buffers[outputIndex3] = (void *) prob3;
#endif
    
    CHECK(cudaStreamCreate(&stream));
    scale = std::min(INPUT_W / (img_w*1.0), INPUT_H / (img_h*1.0));    
    int unpad_w = scale * img_w;
    int unpad_h = scale * img_h;
    re = cv::Mat(unpad_h, unpad_w, CV_8UC3);
    std::cout<<unpad_h<<" "<< unpad_w<<endl;
    int data_idx = 0;
    for (int i = 0; i < INPUT_H; ++i)
    {
        // uchar* pixel = img.ptr<uchar>(i);  // point to first color in row
        for (int j = 0; j < INPUT_W; ++j)
        {
            blob[data_idx] = 114.0f/255.f;
            blob[data_idx+INPUT_H*INPUT_W] = 114.0f/255.f;
            blob[data_idx+2*INPUT_H*INPUT_W] = 114.0f/255.f;
            data_idx++;
        }
    }
    // warm-up
    for (int i=0; i<20;i++)
        context->executeV2(buffers.data());


}
void Yolo7::detect(const cv::Mat &img,std::vector<Object> &objects)
{
    int img_w = img.cols;
    int img_h = img.rows;
    cv::resize(img, re, re.size());
    blobFromImage2(re,blob);

#ifdef CLASSIC_MEM_
        CHECK(cudaMemcpyAsync(buffers[inputIndex], blob, 3 * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyHostToDevice, stream));
#endif

#ifdef UNIFIED_MEM_
        cudaStreamAttachMemAsync(stream,blob,0,cudaMemAttachGlobal);
#endif
    
    context->enqueueV2(buffers.data(), stream, nullptr);

#ifdef CLASSIC_MEM_       
        CHECK(cudaMemcpyAsync(prob1, buffers[outputIndex1], output_size1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaMemcpyAsync(prob2, buffers[outputIndex2], output_size2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaMemcpyAsync(prob3, buffers[outputIndex3], output_size3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
#endif        
        
#ifdef UNIFIED_MEM_
        cudaStreamSynchronize(stream); // this additional bcuse of Note on page 10 cuda for tegra doc 
        cudaStreamAttachMemAsync(stream,prob1,0,cudaMemAttachHost);
        cudaStreamAttachMemAsync(stream,prob2,0,cudaMemAttachHost);
        cudaStreamAttachMemAsync(stream,prob3,0,cudaMemAttachHost);
#endif

    cudaStreamSynchronize(stream);
    decode_outputs(prob1,prob2,prob3, objects, scale, img_w, img_h);

#ifdef UNIFIED_MEM_
        cudaStreamAttachMemAsync(stream,blob,0,cudaMemAttachHost);
        cudaStreamAttachMemAsync(stream,prob1,0,cudaMemAttachGlobal);
        cudaStreamAttachMemAsync(stream,prob2,0,cudaMemAttachGlobal);
        cudaStreamAttachMemAsync(stream,prob3,0,cudaMemAttachGlobal);
#endif

}

void Yolo7::generete_proposal_scale(float* feat_blob, float prob_threshold, std::vector<Object>& objects,int nx, int ny,float stride,float * anchor_grid_w,float * anchor_grid_h)
{
    
    for(int i1=0;i1<3;i1++)
    for(int i2=0;i2<ny;i2++) // nx = 80 stride = 8 
    for(int i3=0;i3<nx;i3++) // ny = 48
    {
        int i4 = 0;
        int this_index = i3*85 + i2*85*nx + i1*85*nx*ny;
        float x = feat_blob[i4+this_index];
        x = (x*2-0.5+i3)*stride ; 
        i4++;
        float y = feat_blob[i4+this_index];
        y = (y*2-0.5+i2)*stride; 
        i4++;
        float w = feat_blob[i4+this_index];
        w = w*2*w*2*anchor_grid_w[i1];
        i4++;
        float h = feat_blob[i4+this_index];
        h = h*2*h*2*anchor_grid_h[i1];
        i4++;
        float obj_conf = feat_blob[i4+this_index];
        i4++;
        
        if (obj_conf > prob_threshold)
        {
            std::array<float, NUM_CLASSES> &classes_scores = reinterpret_cast<std::array<float, NUM_CLASSES>&>(feat_blob[i4+this_index]);

            auto max_score = std::max_element(classes_scores.begin() , classes_scores.end()); 
            int argmaxVal = distance(classes_scores.begin(), max_score);

            float prob = classes_scores[argmaxVal]*obj_conf;

            Object obj;
            obj.prob = prob;
            obj.label = argmaxVal;
            float x0 = x - w * 0.5f;
            float y0 = y - h * 0.5f;
            obj.rect = cv::Rect_<float>(x0,y0,w,h);
            objects.emplace_back(obj);
        }
    }
}


void Yolo7::decode_outputs(float* prob1,float* prob2,float* prob3, std::vector<Object>& objects, float scale, const int img_w, const int img_h) {
        std::vector<Object> proposals;

        generete_proposal_scale(prob1,BBOX_CONF_THRESH,proposals,80,48,8,anchor_grid_w_1,anchor_grid_h_1);
        generete_proposal_scale(prob2,BBOX_CONF_THRESH,proposals,40,24,16,anchor_grid_w_2,anchor_grid_h_2);
        generete_proposal_scale(prob3,BBOX_CONF_THRESH,proposals,20,12,32,anchor_grid_w_3,anchor_grid_h_3);
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);
        int count = picked.size();
        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];
            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x) / scale;
            float y0 = (objects[i].rect.y) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;
            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }
}

Yolo7::~Yolo7()
{
    CHECK(cudaFree(buffers[inputIndex]));

    CHECK(cudaFree(buffers[outputIndex1]));
    CHECK(cudaFree(buffers[outputIndex2]));
    CHECK(cudaFree(buffers[outputIndex3]));
    cudaStreamDestroy(stream);
    delete[] blob;
    delete[] prob1;
    delete[] prob2;
    delete[] prob3;
}




