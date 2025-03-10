
#ifndef __ZJVIDEO_FUNCTION_H__
#define __ZJVIDEO_FUNCTION_H__

#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include "ExtraData.h"

#define ZJ_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ZJ_MIN(a, b) ((a) < (b) ? (a) : (b))


namespace ZJVIDEO
{
    // 通道重命名
    static std::string join_string(std::string str, int i)
    {
        return str + "_c" + std::to_string(i);
    }
    // 重命名后的通道解析, -1 表示不区分通道
    static int parse_id(std::string str)
    {
        std::size_t pos = str.rfind("_c");
        if (pos != std::string::npos)
        {
            std::string i_str = str.substr(pos + 2); // 2 is the length of "_c"
            if (std::all_of(i_str.begin(), i_str.end(), ::isdigit))
            {
                return std::stoi(i_str);
            }
            else
            {
                return -1;
            }
        }
        else
        {
            // Handle error: the string does not contain "_c"
            return -1;
        }
    }

    // 这个函数返回a除以b的结果，向上取整。如果a不能被b整除，那么结果会加1。这个函数常用于计算需要多少个大小为b的块才能完全覆盖大小为a的区域。
    template <typename T>
    static inline T uDivUp(T a, T b)
    {
        return (a / b) + (a % b != 0);
    }
    // 这个函数返回最小的能被b整除的数，且这个数不小于a。这个函数常用于将a向上调整到最近的b的倍数。
    template <typename T>
    static inline T uSnapUp(T a, T b)
    {
        return (a + (b - a % b) % b);
    }
    // 这个函数返回最大的能被b整除的数，且这个数不大于a。这个函数常用于将a向下调整到最近的b的倍数。
    template <typename T>
    static inline T uSnapDown(T a, T b)
    {
        return (a - (a % b));
    }
    // 这个函数返回a到最近的能被b整除的数的距离。这个函数常用于计算a到最近的b的倍数的距离。
    template <typename T>
    static inline T uSnapDelta(T a, T b)
    {
        return (b - a % b) % b;
    }

    static float IoU(const DetectBox &a, const DetectBox &b)
    {
        float interArea = ZJ_MAX(0.0f, ZJ_MIN(a.x2, b.x2) - ZJ_MAX(a.x1, b.x1)) *
                          ZJ_MAX(0.0f, ZJ_MIN(a.y2, b.y2) - ZJ_MAX(a.y1, b.y1));
        float unionArea = (a.x2 - a.x1) * (a.y2 - a.y1) +
                          (b.x2 - b.x1) * (b.y2 - b.y1) -
                          interArea;
        return interArea / unionArea;
    }

    static void NMS(std::vector<DetectBox> &boxes, float nms_thresh)
    {
        std::sort(boxes.begin(), boxes.end(), [](const DetectBox &a, const DetectBox &b)
                  { return a.score > b.score; });

        for (size_t i = 0; i < boxes.size(); ++i)
        {
            if (boxes[i].score == 0)
                continue;
            for (size_t j = i + 1; j < boxes.size(); ++j)
            {
                if (IoU(boxes[i], boxes[j]) > nms_thresh)
                {
                    boxes[j].score = 0;
                }
            }
        }

        boxes.erase(std::remove_if(boxes.begin(), boxes.end(), [](const DetectBox &a)
                                   { return a.score == 0; }),
                    boxes.end());
    }

    static void softmax(float *output, int num)
    {
        float max_output = *std::max_element(output, output + num);
        float sum = 0.0f;

        for (int i = 0; i < num; i++)
        {
            output[i] = std::exp(output[i] - max_output);
            sum += output[i];
        }

        for (int i = 0; i < num; i++)
        {
            output[i] /= sum;
        }
    }

    // 一维数组, vector中存放float的指针地址，仅处理单个数据
    static void softmax(std::vector<float *> output)
    {
        size_t num = output.size();
        
        float max_output = -std::numeric_limits<float>::infinity();
        for(int i = 0; i < num; i++)
        {
            max_output = ZJ_MAX(max_output, *(output[i]));
        }

        float sum = 0.0f;

        for (int i = 0; i < num; i++)
        {
            *(output[i]) = std::exp(*(output[i]) - max_output);
            sum += *(output[i]);
        }

        for (int i = 0; i < num; i++)
        {
            *(output[i]) /= sum;
        }
    }

} // namespace ZJVIDEO

#endif // __ZJVIDEO_FUNCTION_H__