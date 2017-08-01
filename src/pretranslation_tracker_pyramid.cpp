#include "pretranslation_tracker_pyramid.h"
#include "models.h"
#include "channels.h"
#include "utils.h"

using namespace cv;
using namespace std;
template <class M, class C>
using PTP = PretranslationTrackerPyramid<M, C>;

template <class M, class C>
PretranslationTrackerPyramid<M, C>::PretranslationTrackerPyramid(AlgorithmParameters ap) :
  _levels(ap.levels)
{
  _tracker_pyr.clear();
  _pre_tracker_pyr.clear();
  _tf_pyr.clear();
  _wI_pyr.clear();
  _wM_pyr.clear();
  AlgorithmParameters tr_ap = ap;
  //tr_ap.min_disp /= 2;
  for (int i = 0; i < _levels; i++) {
    _tracker_pyr.push_back(new Tracker<M, C>(ap));
    _pre_tracker_pyr.push_back(new Tracker<Translation, C>(ap));

    _wI_pyr.push_back(Mat());
    _wM_pyr.push_back(Mat());
    _tf_pyr.push_back(
      [&,i](const cv::Mat& I, const cv::Mat& mask,
            const typename M::ParameterStruct init_params) {
        warp_image<M>(I, init_params, _wI_pyr[i]);
        warp_image<M>(mask, init_params, _wM_pyr[i]);
        Result<Translation> tres = _pre_tracker_pyr[i]->step(_wI_pyr[i], _wM_pyr[i],
                                                           Translation::ParameterStruct());
        typename M::ParameterStruct translated_params = init_params;
        for (int j = 0; j < M::DOF/2; j++) {
          translated_params.vec.segment(2*j, 2) += tres.params.vec;
        }
        Result<M> res = _tracker_pyr[i]->step(I, mask, translated_params);
        
        return compose_results<Translation, M>(tres, res);
      });
  }

  _tracker = GeneralTrackerPyramid<M>(_tf_pyr);

}
template <class M, class C>
PretranslationTrackerPyramid<M, C>::~PretranslationTrackerPyramid() {
  for (int i = 0; i < _levels; i++) {
    delete _tracker_pyr[i];
    delete _pre_tracker_pyr[i];
  }
}
template <class M, class C>
void PretranslationTrackerPyramid<M, C>::set_template(const Mat& I, const PointVec& pts,
                             const typename M::ParameterStruct& identity_params)
{
  typename M::ParameterStruct params = identity_params, next_params;
  PointVec pts0 = pts, pts1;
  int num_pts = pts0.size();
  Mat I0;
  I.copyTo(I0);
  for (int i = _levels-1; i >= 0; i--) {
    _tracker_pyr[i]->set_template(I0, pts0, params);
    _pre_tracker_pyr[i]->set_template(I0, pts0, Translation::ParameterStruct());
    pyrDown(I0, I0);
    pt_pyrDown(pts0, I0.size(), pts1);
    pts0.swap(pts1);
    M::dec_params(params, 1, next_params);
    params = next_params;
  }
}
template <class M, class C>
Result<M> PretranslationTrackerPyramid<M, C>::step(const Mat& I, const Mat& mask,
                          const typename M::ParameterStruct& init_params) {
  return _tracker.step(I, mask, init_params);
}

template class PretranslationTrackerPyramid<Affine, TwoBit>;
template class PretranslationTrackerPyramid<Homography, TwoBit>;

template class PretranslationTrackerPyramid<Affine, OneBit>;
template class PretranslationTrackerPyramid<Homography, OneBit>;
