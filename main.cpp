#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "BYTETracker.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <string>

#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>

#include <iostream>
#include <fstream>
#include <string>
#include <array>

#include "yolo7.hpp"

using namespace std;

#include <fstream>
#include <string>
#include <algorithm>




#define CLASSIC_MEM
// #define UNIFIED_MEM


int main(int argc, char** argv) {
    // cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    // saveEngineFile(argv[1],"../yolo-tiny.engine");
    // return -1;

    if (argc == 4 && std::string(argv[2]) == "-i") {
        const std::string engine_file_path {argv[1]};
        std::ifstream file(engine_file_path, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "run 'python3 yolox/deploy/trt.py -n yolox-{tiny, s, m, l, x}' to serialize model first!" << std::endl;
        std::cerr << "Then use the following command:" << std::endl;
        std::cerr << "./yolox ../model_trt.engine -i ../../../assets/dog.jpg  // deserialize file and run inference" << std::endl;
        return -1;
    }
    const std::string input_image_path {argv[3]};
    cv::VideoCapture cap(input_image_path);
    
    cv::Mat img;
    cap >> img;
    int fps = cap.get(CAP_PROP_FPS);
        // stop the program if no more images
    if(img.rows==0 || img.cols==0)
        return -1;
    int img_w = img.cols;
    int img_h = img.rows;

    Yolo7 yolo7(argv[1],img_w,img_h);
    BYTETracker tracker(fps, 30);


    for(;;)
    {
        cap >> img;
        // stop the program if no more images
        if(img.rows==0 || img.cols==0)
            break;

        int img_w = img.cols;
        int img_h = img.rows;
        std::vector<Object> objects;

        auto start = std::chrono::system_clock::now();        

        yolo7.detect(img,objects);
        vector<STrack> output_stracks = tracker.update(objects);
        
        auto end = std::chrono::system_clock::now();
        auto micro= std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout<<1e6/micro<<" FPS"<<std::endl;

        for (int i = 0; i < output_stracks.size(); i++)
		{
			vector<float> tlwh = output_stracks[i].tlwh;
			bool vertical = tlwh[2] / tlwh[3] > 1.6;
			// if (tlwh[2] * tlwh[3] > 20 && !vertical)
			{
				Scalar s = tracker.get_color(output_stracks[i].track_id);
				putText(img, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5), 
                        0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
                rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
			}
		}

        cv::imshow("frame",img);

        int key = cv::waitKey(1); 
            if(key == 'q') break;
    }


    return 0;
}






