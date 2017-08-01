#ifndef BP_GEN_TRACKER_PYR_H
#define BP_GEN_TRACKER_PYR_H
#include "types.h"
#include <vector>

template <class Model>
class GeneralTrackerPyramid {
public:
  GeneralTrackerPyramid() {}
  GeneralTrackerPyramid(std::vector< typename TrackFunction<Model>::T > tf_pyr);
  Result<Model> step(const cv::Mat& I, const cv::Mat& mask,
                     const typename Model::ParameterStruct& init_params);
private:
  std::vector< typename TrackFunction<Model>::T > _tf_pyr;
  std::vector< cv::Mat > _img_pyr, _mask_pyr;
  int _levels;
};
#endif
