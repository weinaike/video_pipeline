
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

void amplify(cil::CImg<unsigned char>& img, int factor) {
    cimg_forXY(img, x, y) {
        for (int c = 0; c < img.spectrum(); ++c) {
            img(x, y, 0, c) = std::min(255, img(x, y, 0, c) * factor);
        }
    }
}
void HSVtoRGB(float h, float s, float v, float& r, float& g, float& b) {
    int i = h * 6;
    float f = h * 6 - i;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);
    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }
}

cil::CImg<unsigned char> create_color_mask(const cil::CImg<unsigned char>& mask, int num_classes) 
{
    cil::CImg<unsigned char> color_mask(mask.width(), mask.height(), 1, 3, 0);
    cimg_forXY(mask, x, y) 
    {
        int label = mask(x, y);
        // 在HSV颜色空间中平均分配颜色
        float hue =  1.0f * label / num_classes;
        float saturation = 1.0f;
        float value = 1.0f;
        // 将HSV颜色转换为RGB颜色
        float r, g, b;
        HSVtoRGB(hue, saturation, value, r, g, b);
        color_mask(x, y, 0, 0) = r * 255;
        color_mask(x, y, 0, 1) = g * 255;
        color_mask(x, y, 0, 2) = b * 255;
    }
    return color_mask;
}

// 将彩色的mask叠加到原图上
void overlay_mask(cil::CImg<unsigned char>& img, const cil::CImg<unsigned char>& color_mask) 
{
    img.draw_image(0, 0, 0, 0, color_mask, 0.1f);
}

int input_worker(std::function<int(const std::shared_ptr<ZJVIDEO::FrameData> & )> func, int camera_id)
{
    
    int cnt = 0;
    cil::CImg<unsigned char> img(pic_path.c_str());
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

    // std::string cfg_file = "../configure/pipeline_sample_segment.json";
    std::string cfg_file = "../configure/pipeline_sample_infer.json";
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
    cil::CImg<unsigned char> img(pic_path.c_str());
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
            else if(data->data_name == "SegmentResult")
            {
                std::shared_ptr<const ZJVIDEO::SegmentResultData> result = std::dynamic_pointer_cast<const ZJVIDEO::SegmentResultData>(data);
                cil::CImg<unsigned char> mask(result->mask->width, result->mask->height, 1, 1, 0);
                memcpy(mask.data(), result->mask->data->cpu_data(), mask.size()*sizeof(unsigned char));
                cil::CImg<unsigned char> color_mask = create_color_mask(mask, 20);
                overlay_mask(img, color_mask);

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