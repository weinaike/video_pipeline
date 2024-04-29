#include "PreProcessor.h"

#define CIMG_DEBUG 0

#if CIMG_DEBUG
#include "CImg/CImg.h"
#endif


#define Enable_CUDA
#ifdef Enable_CUDA
#include "cuda_kernels/CudaPreProcess.h"
#endif

namespace ZJVIDEO
{
#ifdef Enable_CUDA

    static void invertAffineTransform(float matM[2][3], float iM[2][3])
    {
        float D = matM[0][0]*matM[1][1] - matM[0][1]*matM[1][0];
        D = D != 0. ? 1./D : 0.;
        float A11 = matM[1][1]*D, A22 = matM[0][0]*D, A12 = -matM[0][1]*D, A21 = -matM[1][0]*D;
        float b1 = -A11*matM[0][2] - A12*matM[1][2];
        float b2 = -A21*matM[0][2] - A22*matM[1][2];

        iM[0][0] = A11; iM[0][1] = A12; iM[0][2] = b1;
        iM[1][0] = A21; iM[1][1] = A22; iM[1][2] = b2;
    }


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
            frame_data->data->set_device_id(m_device_id);
            unsigned char *data = (unsigned char *)frame_data->data->mutable_gpu_data();
            Rect roi = frame_roi->roi;

            std::vector<int> shape;
            shape.push_back(frame_data->height);
            shape.push_back(frame_data->width);
            shape.push_back(frame_data->channel());
            U8Blob roi_img_blob(shape); // 转为rgb模式
            roi_img_blob.set_device_id(m_device_id);
            unsigned char *roi_img_data = (unsigned char *)roi_img_blob.mutable_gpu_data();
            // std::cout<<shape[0]<<" "<<shape[1]<<" "<<shape[2]<<std::endl;

            int format;
            if (frame_data->format == ZJV_IMAGEFORMAT_RGB24)
            {
                format = CUDA::ColorFormat::RGB;
            }
            else if (frame_data->format == ZJV_IMAGEFORMAT_RGBP)
            {
                format = CUDA::ColorFormat::RGBP;
            }
            else if (frame_data->format == ZJV_IMAGEFORMAT_BGR24)
            {
                format = CUDA::ColorFormat::BGR;
            }
            else
            {
                CLOG(ERROR, PRELOG) << "frame_data format not supported now, only support RGB24 and PRGB24";
                assert(0);
            }
            // crop and cvtcolor to rgb format
            CUDA::crop_cvtcolor_Invoker(data, frame_data->width * frame_data->channel(),
                                  frame_data->width, frame_data->height, roi_img_data,
                                  roi.x, roi.y, roi.width, roi.height, format, NULL);
            // std::cout<<roi.x<<" "<<roi.y<<" "<<roi.width<<" "<<roi.height<<std::endl;


            #if  CIMG_DEBUG
            if(i==0)
            {
                std::cout<<i<<std::endl;
                cil::CImg<unsigned char> roi_img(3,roi.width, roi.height, 1);
                cudaMemcpy(roi_img.data(), roi_img_data, roi_img.size(),cudaMemcpyDeviceToHost);
                roi_img.permute_axes("yzcx");

                cil::CImgDisplay disp1(roi_img,"ROIImage");
                while (!disp1.is_closed()) 
                {
                    disp1.wait();
                    if (disp1.is_key()) {
                        std::cout << "Key pressed: " << disp1.key() << std::endl;
                    }
                }
            }
            #endif

            // 4. 缩放

            float matrix_2_3[2][3];
            matrix_2_3[0][0] = frame_rois[i]->scale_x;
            matrix_2_3[0][1] = 0;
            matrix_2_3[0][2] = frame_rois[i]->padx;
            matrix_2_3[1][0] = 0;
            matrix_2_3[1][1] = frame_rois[i]->scale_y;
            matrix_2_3[1][2] = frame_rois[i]->pady;
            
            float matrix_2_3_inv[2][3] = {0};
            invertAffineTransform(matrix_2_3, matrix_2_3_inv);

            // 打印matrix_2_3
            #if CIMG_DEBUG
            for(int t = 0; t < 6; t++)
            {
                int s = t / 3;
                CLOG(INFO, PRELOG) << "matrix_2_3[" << t << "] = " << matrix_2_3_inv[s][t%3];
            }
            #endif
            // // 为matrix_2_3 申请cuda设备内存
            float *matrix_2_3_device = nullptr;
            cudaMalloc(&matrix_2_3_device, 6 * sizeof(float));
            cudaMemcpy(matrix_2_3_device, matrix_2_3_inv, 6 * sizeof(float), cudaMemcpyHostToDevice);

            unsigned char pad_value =  param.letterbox_value[0];
            CUDA::Norm norm;
            
            float alpha = 1.0f;
            float mean[3] = {param.mean_value[0], param.mean_value[1], param.mean_value[2]};    // [0-255]
            float std[3] = {param.std_value[0], param.std_value[1], param.std_value[2]};        // [0-255]
            // out = (x - mean) / std * alpha
            norm = CUDA::Norm::mean_std(mean, std, alpha);


            CUDA::warpAffineBilinearAndNormalizePlaneInvoker(roi_img_data, roi.width * 3,
                                                    roi.width, roi.height, input_data + count* i,
                                                    param.resize_width, param.resize_height, matrix_2_3_device, pad_value,
                                                    norm, NULL);
            cudaFree(matrix_2_3_device);          
            
            #if CIMG_DEBUG
                if(i == 0)
                {

                    // 打印norm
                    for(int t = 0; t < 3; t++)
                    {
                        CLOG(INFO, PRELOG) << "norm.mean[" << t << "] = " << norm.mean[t];
                        CLOG(INFO, PRELOG) << "norm.std[" << t << "] = " << norm.std[t];
                    }
                    // 打印norm.alpha
                    CLOG(INFO, PRELOG) << "norm.alpha = " << norm.alpha;
                    
                    std::cout<<roi.width << " "<<roi.height << " " << count << " "<< i << " " <<pad_value << std::endl;

                    std::cout<<param.resize_width<<" "<<param.resize_height<<" "<<param.resize_channel<<std::endl;
                    cil::CImg<float> img(param.resize_width, param.resize_height, 1, param.resize_channel);
                    cudaMemcpy(img.data(), input_data + count* i, count * sizeof(float), cudaMemcpyDeviceToHost);

                    cil::CImgDisplay disp(img,"After Resize");
                    while (!disp.is_closed()) 
                    {
                        disp.wait();
                        if (disp.is_key()) {
                            std::cout << "Key pressed: " << disp.key() << std::endl;
                        }
                    }
                }

            #endif
            // CLOG(INFO, PRELOG) << "PreProcessor::run_cuda: " << "frame_roi: " << i << " done";

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


        
        #if 0//CIMG_DEBUG

        std::vector<int > shape = blob.shape();

        int bs = shape[0];
        int t = shape[1];
        int c = shape[2];
        int h = shape[3];
        int w = shape[4];
        int area = c * h * w;
        const float * cpu = blob.cpu_data();

        // for(int i = 0; i < bs * t; i++)
        for(int i = 0; i < 1; i++)
        {
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
#endif

} // namespace ZJVIDEO {
