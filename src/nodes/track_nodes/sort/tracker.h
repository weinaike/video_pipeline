#ifndef __ZJV_TRACKER_H__
#define __ZJV_TRACKER_H__

#include <map>
#include <vector>
// #include <opencv2/core.hpp>

#include "track.h"
#include "munkres.h"

namespace ZJVIDEO
{

    class Tracker
    {
    public:
        Tracker(int max_coast_cycles = 5, float iou_threshold = 0.3);
        ~Tracker() = default;

        void Run(const std::vector<Rect> &detections);
        std::map<int, Track> GetTracks();

    protected:
        float CalculateIou(const Rect &det, const Track &track);

        void HungarianMatching(const std::vector<std::vector<float>> &iou_matrix,
                                      size_t nrows, size_t ncols,
                                      std::vector<std::vector<float>> &association);

        /**
         * Assigns detections to tracked object (both represented as bounding boxes)
         * Returns 2 lists of matches, unmatched_detections
         * @param detection
         * @param tracks
         * @param matched
         * @param unmatched_det
         * @param iou_threshold
         */
        void AssociateDetectionsToTrackers(const std::vector<Rect> &detection,
                                                  std::map<int, Track> &tracks,
                                                  std::map<int, Rect> &matched,
                                                  std::vector<Rect> &unmatched_det,
                                                  float iou_threshold = 0.3);
    private:
        // Hash-map between ID and corresponding tracker
        std::map<int, Track> tracks_;

        // Assigned ID for each bounding box
        int id_;
        int kMaxCoastCycles_;
        float kIOUthreshold_;
    };

} // namespace ZJVIDEO
#endif //__ZJV_TRACKER_H__
