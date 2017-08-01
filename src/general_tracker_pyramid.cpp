#include "general_tracker_pyramid.h"
#include "models.h"
#include "utils.h"
#include <iostream>

using namespace cv;
using namespace std;


template <class M>
GeneralTrackerPyramid<M>::GeneralTrackerPyramid(vector< typename TrackFunction<M>::T > tf_pyr) :
  _tf_pyr(tf_pyr), _levels(tf_pyr.size())
{
  _img_pyr.resize(_levels);
  _mask_pyr.resize(_levels);
}

template <class M>
Result <M> GeneralTrackerPyramid<M>::step(const cv::Mat& I, const cv::Mat& mask,
                                          const typename M::ParameterStruct& init_params) {

  Result <M> r;
  typename M::ParameterStruct params;
  M::dec_params(init_params, _levels-1, params);
  Mat I0, M0;
  I.copyTo(I0);
  mask.copyTo(M0);

  for (int i = _levels-1; i >= 0; i--) {
    I0.copyTo(_img_pyr[i]);
    M0.copyTo(_mask_pyr[i]);
    pyrDown(I0, I0);
    pyrDown(M0, M0);
  }

  Result<M> sub_r;
  r.iterations = 0;
  for (int i = 0; i < _levels; i++) {
    sub_r = _tf_pyr[i](_img_pyr[i], _mask_pyr[i], params);
    M::inc_params(sub_r.params, 1, params);
    r = compose_results<M, M>(r, sub_r);
  }
  return r;
}

template class GeneralTrackerPyramid<Translation>;
template class GeneralTrackerPyramid<Affine>;
template class GeneralTrackerPyramid<Homography>;
