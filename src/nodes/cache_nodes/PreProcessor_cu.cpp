#include "PreProcessor.h"

#define CIMG_DEBUG 0

#if CIMG_DEBUG
#include "CImg/CImg.h"
#endif


// #define Enable_CUDA
#ifdef Enable_CUDA
#include "cuda_kernels/CudaPreProcess.h"
#include "cuda_util.cpp"
#endif

namespace ZJVIDEO
{
#ifdef Enable_CUDA

    int PreProcessor::run_cuda(const std::vector<std::shared_ptr<FrameROI>> &frame_rois, FBlob &blob, PreProcessParameter &param)
    {

        blob.set_device_id(m_device_id);
        float * input_data = blob.mutable_gpu_data();
        assert(input_data != nullptr);
        int count = param.resize_channel * param.resize_height * param.resize_width;
        // std::cout<<"count: "<<count<<" all count "<< blob.count()<<std::endl;

        for (int i = 0; i < frame_rois.size(); i++)
        {
            // 1. 提取图片
            const std::shared_ptr<FrameROI> frame_roi = frame_rois[i];
            std::shared_ptr<const FrameData> frame_data = frame_roi->frame;
            Rect roi = frame_roi->roi;

            float matrix_2_3[2][3];
            matrix_2_3[0][0] = frame_rois[i]->scale_x;
            matrix_2_3[0][1] = 0;
            matrix_2_3[0][2] = frame_rois[i]->padx;
            matrix_2_3[1][0] = 0;
            matrix_2_3[1][1] = frame_rois[i]->scale_y;
            matrix_2_3[1][2] = frame_rois[i]->pady;

            cuda_preprocess(frame_data, roi, input_data + count * i, matrix_2_3, param, m_device_id);

        }
        // CLOG(INFO, PRELOG) << "PreProcessor::run_cuda: done";
        
        
        #if 0 //CIMG_DEBUG

            std::vector<int > shape = blob.shape();
            int bs = shape[0];
            int c = shape[1];
            int h = shape[2];
            int w = shape[3];
            int area = c * h * w;
            const float * cpu = blob.cpu_data();
            
            for(int i = 0; i < bs; i++)
            {
                for(int j = 0; j < c; j++)
                {
                    for(int k = 0; k < h; k++)
                    {
                        for(int l = 0; l < w; l++)
                        {
                            int index = i * area + j * h * w + k * w + l;
                            // std::cout<<cpu[index]<<" ";
                        }
                    }
                }
                cil::CImg<float> img(w, h, 1, c);
                memcpy(img.data(), cpu + area * i, area * sizeof(float));
                // img.save("float.jpg");

                cil::CImgDisplay disp(img,"After Resize II");
                while (!disp.is_closed()) 
                {
                    disp.wait();
                    if (disp.is_key()) {
                        std::cout << "Key pressed: " << disp.key() << std::endl;
                    }
                }
            }   
        #endif


        
        #if 0 //CIMG_DEBUG

            std::vector<int > shape = blob.shape();

            int bs = shape[0];
            int t = shape[1];
            int c = shape[2];
            int h = shape[3];
            int w = shape[4];
            int area = c * h * w;
            const float * cpu = blob.cpu_data();

            std::cout<<"\noutput data:"<<std::endl;
            // for(int i = 0; i < bs * t; i++)
            for(int i = 0; i < 1; i++)
            {
                for(int j = 0; j < 1; j++)
                {
                    for(int k = 0; k < 1; k++)
                    {
                        for(int l = 0; l < w; l++)
                        {
                            int index = i * area + j * h * w + k * w + l;
                            std::cout<<cpu[index]<<" ";
                        }
                    }
                }
            }   
        #endif

        return ZJV_STATUS_OK;
    }

    int PreProcessor::run_3d_cuda(const std::vector<std::shared_ptr<FrameROI>> &frame_rois, FBlob &blob, PreProcessParameter &param)
    {
        
        std::vector<std::shared_ptr<FrameROI>> frame_rois_3d;
        for(auto frame_roi : frame_rois)
        {
            for(auto frame : frame_roi->frames)
            {
                std::shared_ptr<FrameROI> frame_roi_3d = std::make_shared<FrameROI>();
                frame_roi_3d->frame = frame;
                frame_roi_3d->roi = frame_roi->roi;
                frame_roi_3d->input_vector_id = frame_roi->input_vector_id;
                frame_roi_3d->original = frame_roi->original;

                frame_roi_3d->input_width = frame_roi->input_width;
                frame_roi_3d->input_height = frame_roi->input_height;

                frame_roi_3d->padx = frame_roi->padx;
                frame_roi_3d->pady = frame_roi->pady;
                frame_roi_3d->scale_x = frame_roi->scale_x;
                frame_roi_3d->scale_y = frame_roi->scale_y;
                frame_roi_3d->resize_type = frame_roi->resize_type;

                frame_roi_3d->pre_process = frame_roi->pre_process;

                frame_rois_3d.push_back(frame_roi_3d);
            }            
        }


        if(param.output_format == ZJV_PREPROCESS_OUTPUT_FORMAT_NCTHW)
        {
            // std::cout<<"run_3d_cuda"<<std::endl;
            std::vector<int> shape = blob.shape();
            int tmp = shape[1];
            shape[1] = shape[2];
            shape[2] = tmp;
            
            FBlob blob_3d(shape);  
            const float * cpu_data = blob.cpu_data();
            
            run_cuda(frame_rois_3d, blob_3d, param);

            float * input_data = blob.mutable_gpu_data();
            const float * input_data_3d = blob_3d.gpu_data();

            CUDA::permute_CT(input_data, input_data_3d, shape[0], shape[1], shape[2], shape[3], shape[4]);

            #if 0
                int N = shape[0];
                int T = shape[1];
                int C = shape[2];
                int H = shape[3];
                int W = shape[4];
                
                int n = 0;
                int t = 1;
                int c = 2;
                int h = 1;
                int w = 1;

                int idx1 = n*(T*C*H*W) + t*(C*H*W) + c*(H*W) + h*W + w;
                int idx2 = n*(C*T*W*C) + c*(T*H*W) + t*(H*W) + h*W + w;

                float f1;
                float f2;
                cudaMemcpy(&f1, &input_data_3d[idx1], sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&f2, & input_data[idx2], sizeof(float), cudaMemcpyDeviceToHost);
                
                std::cout<<"f1: "<<f1<<" f2: "<<f2<<std::endl;
            #endif
        }
        else if(param.output_format == ZJV_PREPROCESS_OUTPUT_FORMAT_NTCHW)
        {
            run_cuda(frame_rois_3d, blob, param);
        }
        return ZJV_STATUS_OK;
    }

#else
    int PreProcessor::run_cuda(const std::vector<std::shared_ptr<FrameROI>> &frame_rois, FBlob &blob, PreProcessParameter &param)
    {
        run_cimg(frame_rois, blob, param);
        return ZJV_STATUS_OK;
    }

    int PreProcessor::run_3d_cuda(const std::vector<std::shared_ptr<FrameROI>> &frame_rois, FBlob &blob, PreProcessParameter &param)
    {
        CLOG(ERROR, PRELOG) << "run_3d_cuda not supported now";
        return ZJV_STATUS_OK;
    }
#endif

} // namespace ZJVIDEO {
