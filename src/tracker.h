#ifndef BP_TRACKER_H
#define BP_TRACKER_H
#include <Eigen/Core>
#include "types.h"
#include <opencv2/opencv.hpp>

template<class Model, class Channel>
class Tracker {
public:
  Tracker(AlgorithmParameters ap);
  void set_template(const cv::Mat& I, const PointVec & pts,
                    const typename Model::ParameterStruct& identity_params);
  Result<Model> step(const cv::Mat& I, const cv::Mat& mask,
                     const typename Model::ParameterStruct& init_params);
private:
  typename Model::Jacobian  _J, _Jweighted;
  typename Model::Hessian _H;
  cv::Mat _channels1, _channels2, _xmap, _ymap, _wI, _wmask, _I;
  int _num_pts, _iterations;
  bool *_debug;
  float _min_disp;
  PointVec _pts, _wpts;
  Eigen::Matrix<float, Eigen::Dynamic, 1> _residuals, _weights;
};
#endif
