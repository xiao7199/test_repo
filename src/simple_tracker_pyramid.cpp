#include "simple_tracker_pyramid.h"
#include "models.h"
#include "channels.h"
#include "utils.h"

using namespace cv;
using namespace std;

template <class M, class C>
SimpleTrackerPyramid<M, C>::SimpleTrackerPyramid(AlgorithmParameters ap) :
  _levels(ap.levels)
{
  _tracker_pyr.clear();
  _tf_pyr.clear();
  AlgorithmParameters tmp = ap;
  for (int i = 0; i < _levels; i++) {
    tmp.min_disp = ap.min_disp/(1 << (_levels-i-1));
    _tracker_pyr.push_back(new Tracker<M, C>(tmp));
    _tf_pyr.push_back([&,i](const cv::Mat& I, const cv::Mat& mask,
                          const typename M::ParameterStruct init_params) {
                        return _tracker_pyr[i]->step(I, mask, init_params);
                      });
  }

  _tracker = GeneralTrackerPyramid<M>(_tf_pyr);

}

template <class M, class C>
SimpleTrackerPyramid<M, C>::~SimpleTrackerPyramid() {
  for (int i = 0; i < _levels; i++) {
    delete _tracker_pyr[i];
  }
}

template <class M, class C>
void SimpleTrackerPyramid<M, C>::set_template(const Mat& I, const PointVec& pts,
                                              const typename M::ParameterStruct& identity_params)
{
  PointVec pts0 = pts, pts1;
  int num_pts = pts0.size();
  Mat I0;
  I.copyTo(I0);
  typename M::ParameterStruct params = identity_params, next_params;
  for (int i = _levels-1; i >= 0; i--) {
    _tracker_pyr[i]->set_template(I0, pts0, params);
    pyrDown(I0, I0);
    pt_pyrDown(pts0, I0.size(), pts1);
    pts0.swap(pts1);
    M::dec_params(params, 1, next_params);
    params = next_params;
  }
}
template <class M, class C>
Result<M> SimpleTrackerPyramid<M, C>::step(const Mat& I, const Mat& mask,
                                           const typename M::ParameterStruct& init_params) {
  
  return _tracker.step(I, mask, init_params);
}

template class SimpleTrackerPyramid<Translation, TwoBit>;
template class SimpleTrackerPyramid<Affine, TwoBit>;
template class SimpleTrackerPyramid<Homography, TwoBit>;

template class SimpleTrackerPyramid<Translation, OneBit>;
template class SimpleTrackerPyramid<Affine, OneBit>;
template class SimpleTrackerPyramid<Homography, OneBit>;
