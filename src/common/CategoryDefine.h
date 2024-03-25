

#ifndef __ZJV_CATEGORYDEFINE_H__
#define __ZJV_CATEGORYDEFINE_H__

namespace ZJVIDEO
{
    enum MAIN_CATEGORY
    {
        ZJV_CATEGORY_UNKNOWN = 0,
        // 人
        ZJV_CATEGORY_PERSON = 1,
        // 交通工具
        ZJV_CATEGORY_VEHICLE = 2,
        // 交通标识
        ZJV_CATEGORY_SIGN = 3,
        // 动物
        ZJV_CATEGORY_ANIMAL = 4,
        // 运动器材
        ZJV_CATEGORY_SPORTS = 5,
        // 食品
        ZJV_CATEGORY_FOOD = 6,
        // 家具
        ZJV_CATEGORY_FURNITURE = 7,
        // 电子产品
        ZJV_CATEGORY_ELECTRONIC = 8,
        // 随身物品
        ZJV_CATEGORY_PERSONALITEMS = 9,
        ZJV_CATEGORY_OTHER = 10,
        ZJV_CATEGORY_MAX
    };
    enum VEHICLE_SUB_CATEGORY
    {
        ZJV_VEHICLE_SUB_CATEGORY_UNKNOWN = 0,
        ZJV_VEHICLE_SUB_CATEGORY_CAR = 1,
        ZJV_VEHICLE_SUB_CATEGORY_TRUCK = 2,
        ZJV_VEHICLE_SUB_CATEGORY_BUS = 3,
        ZJV_VEHICLE_SUB_CATEGORY_VAN = 4,        
        ZJV_VEHICLE_SUB_CATEGORY_MOTORCYCLE = 5,
        ZJV_VEHICLE_SUB_CATEGORY_BICYCLE = 6,
        ZJV_VEHICLE_SUB_CATEGORY_PEDESTRIAN = 7,
        // 飞机
        ZJV_VEHICLE_SUB_CATEGORY_AIRPLANE = 8,
        // 船
        ZJV_VEHICLE_SUB_CATEGORY_SHIP = 9,
        // 火车
        ZJV_VEHICLE_SUB_CATEGORY_TRAIN = 10,
        // 其他
        ZJV_VEHICLE_SUB_CATEGORY_OTHER = 11,
        ZJV_VEHICLE_SUB_CATEGORY_MAX
    };
    enum PERSON_SUB_CATEGORY
    {
        ZJV_PERSON_SUB_CATEGORY_PERSON_UNKNOWN = 0,
        ZJV_PERSON_SUB_CATEGORY_BODY = 1,
        ZJV_PERSON_SUB_CATEGORY_FACE = 2,
        ZJV_PERSON_SUB_CATEGORY_HEAD = 3,
        ZJV_PERSON_SUB_CATEGORY_HAND = 4,
        ZJV_PERSON_SUB_CATEGORY_HEADSHOULDER = 5,
        ZJV_PERSON_SUB_CATEGORY_UPPER_BODY = 6,
        ZJV_PERSON_SUB_CATEGORY_LOWER_BODY = 7,
        ZJV_PERSON_SUB_CATEGORY_OTHER = 8,        
        ZJV_PERSON_SUB_CATEGORY_MAX
    };
}

#endif //__ZJV_CATEGORYDEFINE_H__