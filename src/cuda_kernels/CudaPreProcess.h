//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_PREPROCESS_CUH
#define VIDEOPIPELINE_PREPROCESS_CUH

#include <iostream>
#include <cuda_runtime.h>

namespace CUDA
{
    /// @brief 
    enum  ColorFormat
    {
        RGBP = 0,
        RGB = 1,
        BGR = 2
    };

    enum class NormType : int
    {
        NormNone = 0,
        MeanStd = 1,
        AlphaBeta = 2
    };

    enum class ChannelType : int
    {
        Channel_None = 0,
        Invert = 1
    };

    struct Norm
    {
        float mean[3]{};
        float std[3]{};
        float alpha{}, beta{};
        NormType type = NormType::NormNone;
        ChannelType channel_type = ChannelType::Channel_None;

        // out = (x * alpha - mean) / std
        static Norm mean_std(const float mean[3],
                             const float std[3],
                             float alpha = 1 / 255.0f,
                             ChannelType channel_type = ChannelType::Channel_None);

        // out = x * alpha + beta
        static Norm alpha_beta(float alpha,
                               float beta = 0,
                               ChannelType channel_type = ChannelType::Channel_None);
    };

    /*!
     * cropInvoker
     * @details 裁剪图像
     * @param src 输入图像
     * @param src_line_size 输入图像一行的字节数
     * @param src_width 输入图像宽度
     * @param src_height 输入图像高度
     * @param dst 输出图像
     * @param x 起始x坐标
     * @param y 起始y坐标
     * @param width 宽度
     * @param height 高度
     * @param format 输入格式
     * @param stream cuda流
     */
    void crop_cvtcolor_Invoker(uint8_t *src,
                     int src_line_size,
                     int src_width,
                     int src_height,
                     uint8_t *dst,
                     int x,
                     int y,
                     int width,
                     int height,
                     int format,
                     cudaStream_t stream);
    /*!
     * resizeBilinearAndNormalizeInvoker
     * @details 双线性插值缩放图像并归一化 [0, 255] -> [0, 1]
     * @param src: 输入图像
     * @param src_line_size: 输入图像一行的字节数
     * @param src_width: 输入图像宽度
     * @param src_height: 输入图像高度
     * @param dst: 输出图像
     * @param dst_width: 输出图像宽度
     * @param dst_height: 输出图像高度
     * @param norm: 归一化参数
     * @param stream: cuda流
     */
    void resizeBilinearAndNormalizeInvoker(uint8_t *src,
                                           int src_line_size,
                                           int src_width,
                                           int src_height,
                                           float *dst,
                                           int dst_width,
                                           int dst_height,
                                           const Norm &norm,
                                           cudaStream_t stream);
    /*!
     * warpAffineBilinearAndNormalize Plane
     * @details 仿射变换双线性插值缩放图像并归一化 [0, 255] -> [0, 1]
     * @param src 输入图像
     * @param src_line_size 输入图像一行的字节数
     * @param src_width 输入图像宽度
     * @param src_height 输入图像高度
     * @param dst 输出图像
     * @param dst_width 输出图像宽度
     * @param dst_height 输出图像高度
     * @param matrix_2_3 2x3变换矩阵
     * @param const_value 常量值
     * @param norm 归一化参数
     * @param stream cuda流
     */
    void warpAffineBilinearAndNormalizePlaneInvoker(uint8_t *src,
                                                    int src_line_size,
                                                    int src_width,
                                                    int src_height,
                                                    float *dst,
                                                    int dst_width,
                                                    int dst_height,
                                                    float *matrix_2_3,
                                                    uint8_t const_value,
                                                    const Norm &norm,
                                                    cudaStream_t stream);
    /*!
     * warpAffineBilinearAndNormalize Focus
     * @details 仿射变换双线性插值缩放图像并归一化 [0, 255] -> [0, 1]
     * @param src 输入图像
     * @param src_line_size 输入图像一行的字节数
     * @param src_width 输入图像宽度
     * @param src_height 输入图像高度
     * @param dst 输出图像
     * @param dst_width 输出图像宽度
     * @param dst_height 输出图像高度
     * @param matrix_2_3 2x3变换矩阵
     * @param const_value 常量值
     * @param norm 归一化参数
     * @param stream cuda流
     */
    void warpAffineBilinearAndNormalizeFocusInvoker(uint8_t *src,
                                                    int src_line_size,
                                                    int src_width,
                                                    int src_height,
                                                    float *dst,
                                                    int dst_width,
                                                    int dst_height,
                                                    float *matrix_2_3,
                                                    uint8_t const_value,
                                                    const Norm &norm,
                                                    cudaStream_t stream);

    /*!
     * warpPerspective
     * @details 透视变换
     * @param src 输入图像
     * @param src_line_size 输入图像一行的字节数
     * @param src_width 输入图像宽度
     * @param src_height 输入图像高度
     * @param dst 输出图像
     * @param dst_width 输出图像宽度
     * @param dst_height 输出图像高度
     * @param matrix_3_3 3x3变换矩阵
     * @param const_value 常量值
     * @param norm 归一化参数
     * @param stream cuda流
     */
    void warpPerspectiveInvoker(uint8_t *src,
                                int src_line_size,
                                int src_width,
                                int src_height,
                                float *dst,
                                int dst_width,
                                int dst_height,
                                float *matrix_3_3,
                                uint8_t const_value,
                                const Norm &norm,
                                cudaStream_t stream);

    /*!
     * normFeature
     * @details 归一化特征
     * @param feature_array 特征数组
     * @param num_feature 特征数量
     * @param feature_length 特征长度
     * @param stream cuda流
     */
    void normFeatureInvoker(float *feature_array,
                            int num_feature,
                            int feature_length,
                            cudaStream_t stream);

    /*!
     * convertNV12ToBgr
     * @param y y分量
     * @param uv uv分量
     * @param width 宽度
     * @param height 高度
     * @param linesize 一行的字节数
     * @param dst 输出图像
     * @param stream cuda流
     */
    void convertNV12ToBgrInvoker(const uint8_t *y,
                                 const uint8_t *uv,
                                 int width,
                                 int height,
                                 int linesize,
                                 uint8_t *dst,
                                 cudaStream_t stream);

    /**
     * @brief 将BGR格式的图像转换为灰度图
     * @param src
     * @param dst
     * @param width
     * @param height
     */
    void bgr2grayInvoker(uint8_t *src, float *dst, int width, int height, cudaStream_t stream = 0);

    /**
     * @brief permute_CT，切换CT通道数据
     * @param output 输出数据
     * @param input 输入数据
     * @param N batch size
     * @param C 通道数
     * @param T 时间序列
     * @param H 高度
     * @param W 宽度
    */
    void permute_CT(float* output, const float* input, int N, int C, int T, int H, int W);
    

}
 // namespace CUDA

#endif // VIDEOPIPELINE_PREPROCESS_CUH
