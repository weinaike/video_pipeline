#ifndef __ZJV_TRACK_H__
#define __ZJV_TRACK_H__

// #include <opencv2/core.hpp>
#include "kalman_filter.h"
#include "common/Shape.h"
namespace ZJVIDEO
{

    class Track
    {
    public:
        // Constructor
        Track();

        // Destructor
        ~Track() = default;

        void Init(const Rect &bbox);
        void Predict();
        void Update(const Rect &bbox);
        Rect GetStateAsBbox() const;
        float GetNIS() const;

        int coast_cycles_ = 0, hit_streak_ = 0;

    private:
        Eigen::VectorXd ConvertBboxToObservation(const Rect &bbox) const;
        Rect ConvertStateToBbox(const Eigen::VectorXd &state) const;

        KalmanFilter kf_;
    };

}

#endif //__ZJV_TRACK_H__