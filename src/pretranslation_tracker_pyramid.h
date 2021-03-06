#ifndef BP_PRET_TRACKER_PYR_H
#define BP_PRET_TRACKER_PYR_H
#include "types.h"
#include "tracker.h"
#include "models.h"
#include "general_tracker_pyramid.h"
#include <opencv2/opencv.hpp>

template <class Model, class Channel>
class PretranslationTrackerPyramid {
public:
  PretranslationTrackerPyramid(AlgorithmParameters ap);
  ~PretranslationTrackerPyramid();
  void set_template(const cv::Mat& I, const PointVec& pts,
                    const typename Model::ParameterStruct& identity_params);
  Result<Model> step(const cv::Mat& I, const cv::Mat& mask,
                     const typename Model::ParameterStruct& init_params);

private:
  std::vector< Tracker<Model, Channel > * > _tracker_pyr;
  std::vector< Tracker<Translation, Channel > * > _pre_tracker_pyr;  
  std::vector< typename TrackFunction<Model>::T > _tf_pyr;
  std::vector< cv::Mat > _wI_pyr, _wM_pyr;  
  GeneralTrackerPyramid<Model> _tracker;
  int _levels;
};
#endif
