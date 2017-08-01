#ifndef BP_SIM_TRACKER_PYR_H
#define BP_SIM_TRACKER_PYR_H
#include "types.h"
#include "tracker.h"
#include "general_tracker_pyramid.h"
#include <opencv2/opencv.hpp>

template <class Model, class Channel>
class SimpleTrackerPyramid {
public:
  SimpleTrackerPyramid(AlgorithmParameters ap);
  ~SimpleTrackerPyramid();
  void set_template(const cv::Mat& I, const PointVec& pts,
                    const typename Model::ParameterStruct& identity_params);
  Result<Model> step(const cv::Mat& I, const cv::Mat& mask,
                     const typename Model::ParameterStruct& init_params);

private:
  std::vector< Tracker<Model, Channel > * > _tracker_pyr;
  std::vector< typename TrackFunction<Model>::T > _tf_pyr;

  GeneralTrackerPyramid<Model> _tracker;
  int _levels;
};
#endif
