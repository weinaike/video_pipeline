#include "PreProcessor.h"

#define CIMG_DEBUG 0

#if CIMG_DEBUG
#include "CImg/CImg.h"
#endif


// #define Enable_CUDA
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


    static int cuda_preprocess(std::shared_ptr<const FrameData> frame_data, 
                Rect roi, float * norm_data, float matrix_2_3[2][3] , 
                PreProcessParameter &param, int device_id)
    {
        if(frame_data->channel() != param.resize_channel)
        {
            std::cout<<__FILE__<<__LINE__<< "frame_data channel not match resize_channel" << frame_data->channel() << "!=" << param.resize_channel;
            return -1;
        }

        frame_data->data->set_device_id(device_id);
        unsigned char *data = (unsigned char *)frame_data->data->mutable_gpu_data();      

        std::vector<int> shape;
        shape.push_back(frame_data->height);
        shape.push_back(frame_data->width);
        shape.push_back(frame_data->channel());
        U8Blob roi_img_blob(shape); // 转为rgb模式
        roi_img_blob.set_device_id(device_id);
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
        else if(frame_data->format == ZJV_IMAGEFORMAT_GRAY8)
        {
            format = CUDA::ColorFormat::GRAY;
        }
        else
        {
            std::cout<<__FILE__<<__LINE__ << "frame_data format not supported now, only support RGB24 and PRGB24";
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
        float matrix_2_3_inv[2][3] = {0};
        invertAffineTransform(matrix_2_3, matrix_2_3_inv);

        // 打印matrix_2_3
        #if CIMG_DEBUG
        for(int t = 0; t < 6; t++)
        {
            int s = t / 3;
            std::cout << "matrix_2_3[" << t << "] = " << matrix_2_3_inv[s][t%3];
        }
        #endif
        // // 为matrix_2_3 申请cuda设备内存
        float *matrix_2_3_device = nullptr;
        CUDA_CHECK_RETURN(cudaMalloc(&matrix_2_3_device, 6 * sizeof(float)));
        CUDA_CHECK_RETURN(cudaMemcpy(matrix_2_3_device, matrix_2_3_inv, 6 * sizeof(float), cudaMemcpyHostToDevice));

        unsigned char pad_value =  param.letterbox_value[0];
        CUDA::Norm norm;
        
        float alpha = 1.0f;
        float mean[3] = {0};
        float std[3] = {0};
        if(param.resize_channel == 3)
        {
            mean[0] = param.mean_value[0];
            mean[1] = param.mean_value[1];
            mean[2] = param.mean_value[2];
            std[0] = param.std_value[0];
            std[1] = param.std_value[1];
            std[2] = param.std_value[2];
        }
        else
        {
            mean[0] = param.mean_value[0];
            std[0] = param.std_value[0];
        }

        // out = (x - mean) / std * alpha
        norm = CUDA::Norm::mean_std(mean, std, alpha);

        int channel = frame_data->channel();
        CUDA::warpAffineBilinearAndNormalizePlaneInvoker(roi_img_data, roi.width * channel,
                                                roi.width, roi.height, norm_data,
                                                param.resize_width, param.resize_height, matrix_2_3_device, pad_value,
                                                norm, NULL);
        CUDA_CHECK_RETURN(cudaFree(matrix_2_3_device)); 
        
        #if CIMG_DEBUG
            if(i == 0)
            {

                // 打印norm
                for(int t = 0; t < 3; t++)
                {
                    std::cout << "norm.mean[" << t << "] = " << norm.mean[t];
                    std::cout << "norm.std[" << t << "] = " << norm.std[t];
                }
                // 打印norm.alpha
               std::cout << "norm.alpha = " << norm.alpha;
                
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

        return ZJV_STATUS_OK;
    }
    
    #endif

}//namespace ZJVIDEO