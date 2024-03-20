
#include <iostream>
#include "pipeline/Pipeline.h"
#include <chrono>
#include <thread>
#include <csignal>
#include "CImg/CImg.h"

volatile std::sig_atomic_t flag = 0;
void signalHandler(int signum) {
    flag = 1;
}

std::string pic_path = "../data/cat.bmp";

std::string imagenet_file = "../data/synset.txt";

std::vector<std::pair<std::string, std::string>> load_synset(const std::string& filename) 
{
    std::vector<std::pair<std::string, std::string>> synset;
    std::ifstream file(filename);

    if (file.is_open()) {
        std::string line;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string id;
            std::string label;
            ss >> id; // 读取每行的第一个单词作为ID
            std::getline(ss, label); // 读取剩余的部分作为标签
            synset.push_back({id, label});
        }

        file.close();
    } else {
        std::cout << "Unable to open file: " << filename << std::endl;
    }

    return synset;
}

static const char* coco_labels[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
};

int input_worker(std::function<int(const std::shared_ptr<ZJVIDEO::FrameData> & )> func, int camera_id)
{
    
    int cnt = 0;
    cil::CImg<unsigned char> img("../data/cat.bmp");
    // img.display("My Image");
    // int camera_id = 0;
    while (!flag)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        std::shared_ptr<ZJVIDEO::FrameData> frame= std::make_shared<ZJVIDEO::FrameData>();

        frame->width = img.width();
        frame->width = img.width();
        frame->height = img.height();
        frame->channel = img.spectrum();
        frame->depth = 1;
        frame->format = ZJVIDEO::ZJV_IMAGEFORMAT_PRGB24;
        frame->fps = 25;      
        frame->camera_id = camera_id;
        frame->frame_id = cnt;       
        frame->data = std::make_shared<ZJVIDEO::SyncedMemory>(img.size()); 
        std::memcpy(frame->data->mutable_cpu_data(), img.data(), img.size());
        cnt++;
        func(frame);
        
    }
    
    return 0;
}


int main()
{  
    std::vector<std::pair<std::string, std::string>> synset = load_synset(imagenet_file);

    auto time1 = std::chrono::system_clock::now();
    auto time2 = std::chrono::system_clock::now();

    if (time1 == time2) {
        std::cout << "The times are the same.\n";
    } else {
        std::cout << "The times are different.\n";
    }

    // return 0;


    signal(SIGINT, signalHandler);  
    std::cout<< "Hello, World!\n" ;

    std::string cfg_file = "../configure/pipeline_sample_classification.json";
    // std::string cfg_file = "../configure/pipeline_sample_infer.json";
    // std::string cfg_file = "../configure/pipeline_sample.json";
    ZJVIDEO::Pipeline pipeline(cfg_file);

    std::cout<< "pipeline.init()\n" ;
    pipeline.init();
    std::cout<< "pipeline.start()\n" ;
    pipeline.start();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // std::vector<std::string> src_node_name = pipeline.get_src_node_name();
    // std::vector<std::string> dst_node_name = pipeline.get_dst_node_name();
    
    // 打印源节点数量
    // std::cout<< "src_node_name.size(): " << src_node_name.size()  <<std::endl;


    std::vector<std::thread > threads;
    //
    for(int i = 0; i < 1; i++)
    {
        std::thread t1(input_worker, std::bind(&ZJVIDEO::Pipeline::set_input_data, &pipeline, std::placeholders::_1), i);
        threads.emplace_back(std::move(t1));
    }

    int frame_id = 0;
    cil::CImg<unsigned char> img("../data/cat.bmp");
    unsigned char red[] = { 255, 0, 0 };  
    while(!flag)
    {
        std::vector< std::shared_ptr<const ZJVIDEO::BaseData> >datas;
        datas.clear();
        pipeline.get_output_data(datas);
        for(const auto & data :datas)
        {
            if(data->data_name == "DetectResult")
            {
                std::shared_ptr<const ZJVIDEO::DetectResultData> detect_result = std::dynamic_pointer_cast<const ZJVIDEO::DetectResultData>(data);
                for(int i = 0; i < detect_result->detect_boxes.size(); i++)
                {
                    std::cout << " detect_boxes: " << detect_result->detect_boxes[i].x1 << " " << detect_result->detect_boxes[i].y1
                        << " " << detect_result->detect_boxes[i].x2 << " " << detect_result->detect_boxes[i].y2 << std::endl;
                    img.draw_rectangle(detect_result->detect_boxes[i].x1, detect_result->detect_boxes[i].y1,
                        detect_result->detect_boxes[i].x2, detect_result->detect_boxes[i].y2, red,1.0f, ~0U);
                    img.draw_text(detect_result->detect_boxes[i].x1, detect_result->detect_boxes[i].y1, 
                        coco_labels[detect_result->detect_boxes[i].label], red, 0, 1);
                }
            }
            else if(data->data_name == "ClassifyResult")
            {
                std::shared_ptr<const ZJVIDEO::ClassifyResultData> result = std::dynamic_pointer_cast<const ZJVIDEO::ClassifyResultData>(data);
                int label = result->detect_box_categories[0].label;
                float score = result->detect_box_categories[0].score;              

                std::cout << "cat: " << synset[label].second << "  " << score << std::endl;

                // img.draw_text(50, 50, synset[label].second, red, 0, 1);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        pipeline.show_debug_info();
        img.save("../data/result.bmp");
        // cil::CImgDisplay disp(img,"result");
        // while (!disp.is_closed()) {
        //     disp.wait();
        //     if (disp.is_key()) {
        //         std::cout << "Key pressed: " << disp.key() << std::endl;
        //     }
        // }

    }

    for(auto & t : threads)
    {
        if(t.joinable())   t.join();
    }

    pipeline.stop();


    return 0;
}